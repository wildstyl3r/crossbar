use anyhow::{Context, Error as E, Result};
use candle::backprop::GradStore;
use candle::{DType, Device, Tensor};
use candle_nn::Optimizer;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokenizers::Tokenizer;

use crate::Engine;
use crate::chat;
use crate::experiments::enrich::{AMode, Enricher, EnricherSpec};
use crate::experiments::tasks::{Example, HELDOUT_PAIRS, TRAIN_PAIRS, build_dataset, score, shuffle};

/// Training hyperparameters. Defaults restore the healthy-warmup recipe
/// (gate bias +2 open-start, lr ramp 50, gate floor 0.5) with the v2
/// cascade TBPTT objective (see ENRICHER.md).
pub struct TrainConfig {
    /// Initial gate bias of a fresh enricher (+2.0 => sigmoid ~0.88: starts
    /// open, must learn to close).
    pub gate_bias: f64,
    /// Linear lr ramp length at the start of warmup (0 = constant).
    pub lr_warmup: usize,
    /// Gate floor during warmup. The ST phase anneals 0.6 * floor -> 0.
    pub gate_floor: f64,
    /// Starting sampling temperature of the ST phase, annealed to `gt_end`.
    pub gt_start: f64,
    /// Anneal target of the ST sampling temperature (~greedy deployment).
    pub gt_end: f64,
    /// Relative pre-gate injection penalty: `loss + inj_l2 * mean(
    /// ||W shat||^2 / ||q||^2 )`. Pre-gate so closing the gate cannot dodge
    /// it (M1 lesson a). Healthy band 0.25-1x; 0 disables.
    pub inj_l2: f64,
    /// BPTT truncation horizon (in-graph cascade steps at the sequence end).
    pub tau: usize,
    /// Keep the full KV-path gradient in the training graph (exact; at W=0
    /// the KV path is the ONLY cross-step credit — detaching it captured
    /// just 32% of the gradient in the tau-sweep). Disabling it detaches
    /// keys/values for a smaller graph at a large gradient cost.
    pub kv_grad: bool,
    /// Lockstep batch size per micro-step.
    pub batch: usize,
    /// Number of micro-steps accumulated per optimizer step (memory saver:
    /// batch 4 x accum 2 ~= batch 8 at half the in-graph memory).
    pub accum: usize,
    /// Override the tau-sweep diagnostic grid ("tau:kv,tau:kv"); empty =
    /// default comparison grid.
    pub sweep: String,
    /// Global-norm gradient clip over the enricher parameters.
    pub grad_clip: f64,
    /// Seed for the training-data shuffle, the A-matrix init and the ST
    /// sampling rng.
    pub seed: u64,
    /// Skip the terminal evaluation (smoke runs).
    pub skip_eval: bool,
    /// Enricher variant (v0 full-rank by default; see EnricherSpec).
    pub spec: EnricherSpec,
}

fn encode_answer(tokenizer: &Tokenizer, ex: &Example) -> Result<Vec<u32>> {
    let text = format!("{} <end_of_turn>", ex.answer);
    let ids = tokenizer
        .encode(text.as_str(), false)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    Ok(ids)
}

struct Batch {
    /// [b, seq] token ids (prompt tokens, then answer tokens, then pads).
    input_ids: Tensor,
    /// First answer slot position.
    answer_start: usize,
    /// Target ids [b, answer_len], -100 on pads.
    targets: Tensor,
}

fn make_batches(
    examples: &[Example],
    tokenizer: &Tokenizer,
    chat_template: &str,
    batch_size: usize,
) -> Result<Vec<Batch>> {
    let mut rows: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    for ex in examples {
        let rendered = chat::render(chat_template, None, &ex.prompt)?;
        let prompt_tokens = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let answer_tokens = encode_answer(tokenizer, ex)?;
        rows.push((prompt_tokens, answer_tokens));
    }
    // Bucket by exact prompt length so that (a) answer slots share one
    // answer_start across the batch and (b) RoPE positions match inference
    // exactly (no prompt padding). Prompts are never padded; only answer
    // tails are.
    rows.sort_by_key(|(p, a)| (p.len(), p.len() + a.len()));

    let mut batches = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let prompt_len = rows[i].0.len();
        let mut end = i;
        while end < rows.len() && rows[end].0.len() == prompt_len && end - i < batch_size {
            end += 1;
        }
        let slice = &rows[i..end];
        let b = slice.len();
        let answer_len = slice.iter().map(|(_, a)| a.len()).max().unwrap();
        let seq_len = prompt_len + answer_len;
        let mut flat: Vec<u32> = Vec::with_capacity(b * seq_len);
        let mut targets: Vec<i64> = Vec::with_capacity(b * answer_len);
        for (p, a) in slice {
            let mut row: Vec<u32> = p.clone();
            row.extend(a.iter().copied());
            let mut tgt: Vec<i64> = a.iter().map(|&t| t as i64).collect();
            for _ in a.len()..answer_len {
                row.push(0);
                tgt.push(-100);
            }
            flat.extend(&row);
            targets.extend(tgt);
        }
        batches.push(Batch {
            input_ids: Tensor::from_vec(flat, (b, seq_len), &Device::Cpu)?,
            answer_start: prompt_len,
            targets: Tensor::from_vec(targets, (b, answer_len), &Device::Cpu)?,
        });
        i = end;
    }
    Ok(batches)
}

/// Global-norm gradient clipping over the given parameters. Returns the
/// pre-clip norm (for logging).
fn clip_grads_gradstore(grads: &mut GradStore, vars: &[candle::Var], max_norm: f64) -> Result<f64> {
    let mut sq = 0f64;
    let mut per_var: Vec<(candle::Var, f64)> = Vec::new();
    for var in vars {
        if let Some(g) = grads.get(var) {
            let n = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            per_var.push((var.clone(), n));
            sq += n;
        }
    }
    let norm = sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-12);
        let mut scaled: Vec<(candle::Var, Tensor)> = Vec::new();
        for (var, _) in &per_var {
            let g = grads.get(var).context("grad vanished during clip")?.clone();
            scaled.push((var.clone(), g.affine(scale, 0.0)?));
        }
        for (var, g) in scaled {
            grads.insert(&var, g);
        }
    }
    Ok(norm)
}

/// One cascade micro-step: forward + scaled backward. Returns the detached
/// metrics and the raw grad store (caller accumulates `accum` micro-steps
/// before clipping and stepping).
#[allow(clippy::too_many_arguments)]
fn train_backward(
    engine: &Engine,
    enr: &Enricher,
    batch: &Batch,
    tau: usize,
    kv_grad: bool,
    forced_prob: f64,
    gt: f64,
    rng: &mut StdRng,
    cfg_inj_l2: f64,
    accum: usize,
) -> Result<(f64, f64, f64, f64, GradStore)> {
    let out = engine.cascade_train_forward(
        enr,
        &batch.input_ids,
        &batch.targets,
        batch.answer_start,
        tau,
        kv_grad,
        forced_prob,
        gt,
        rng,
    )?;
    let pen_v = out.pen.to_scalar::<f32>()? as f64;
    let mut loss = out.loss.clone();
    if cfg_inj_l2 > 0.0 {
        loss = (loss + (out.pen * cfg_inj_l2)?)?;
    }
    if accum > 1 {
        // Mean over the accumulation group (per-micro mean-of-means; with
        // equal-length buckets the difference from a global mean is tiny).
        loss = loss.affine(1.0 / accum as f64, 0.0)?;
    }
    let grads = loss.backward()?;
    Ok((
        out.ce_value,
        pen_v,
        out.applied_prompt,
        out.applied_answer,
        grads,
    ))
}

/// Merge one micro-step's enricher grads into the accumulator.
fn accumulate_grads(
    acc: &mut std::collections::HashMap<candle::TensorId, candle::Tensor>,
    grads: &GradStore,
    vars: &[candle::Var],
) -> Result<()> {
    use std::collections::hash_map::Entry;
    for var in vars {
        if let Some(g) = grads.get(var) {
            match acc.entry(var.id()) {
                Entry::Occupied(mut e) => {
                    let merged = e.get().add(g)?;
                    e.insert(merged);
                }
                Entry::Vacant(e) => {
                    e.insert(g.clone());
                }
            }
        }
    }
    Ok(())
}

/// Clip + optimizer step over the accumulated grads.
fn accum_step(
    acc: &mut std::collections::HashMap<candle::TensorId, candle::Tensor>,
    vars: &[candle::Var],
    grad_clip: f64,
    opt: &mut candle_nn::AdamW,
) -> Result<f64> {
    let mut merged = GradStore::default();
    for var in vars {
        if let Some(g) = acc.remove(&var.id()) {
            merged.insert_id(var.id(), g);
        }
    }
    let gnorm = clip_grads_gradstore(&mut merged, vars, grad_clip)?;
    opt.step(&merged)?;
    Ok(gnorm)
}

/// TBPTP truncation + KV-path diagnostic: backward the same batch under a
/// grid of (tau, kv_grad) configs and compare the per-tensor gradient norms
/// against the first (reference) config. If a cheaper config's gradient is
/// >= 99% of the reference per tensor, it captures the gradient (see
/// > ENRICHER.md).
fn tau_sweep(
    engine: &Engine,
    enr: &Enricher,
    batch: &Batch,
    configs: &[(usize, bool)],
    inj_l2: f64,
) -> Result<()> {
    println!("tau-sweep: per-tensor grad norms on batch[0] (same data, no optimizer step)");
    let mut rng = StdRng::seed_from_u64(4242);
    let mut rows: Vec<(usize, bool, f64, Vec<f64>)> = Vec::new();
    for &(tau, kv_grad) in configs {
        let t0 = std::time::Instant::now();
        let out = engine.cascade_train_forward(
            enr,
            &batch.input_ids,
            &batch.targets,
            batch.answer_start,
            tau,
            kv_grad,
            1.0,
            1.0,
            &mut rng,
        )?;
        let t1 = std::time::Instant::now();
        let mut loss = out.loss.clone();
        if inj_l2 > 0.0 {
            loss = (loss + (out.pen * inj_l2)?)?;
        }
        let grads = loss.backward()?;
        let t2 = std::time::Instant::now();
        eprintln!(
            "  [timing] tau={tau} kv_grad={kv_grad}: fwd {:?}, bwd {:?} (rss {:.0} MiB)",
            t1 - t0,
            t2 - t1,
            rss_mib()
        );
        // Named per-tensor norms (named_params() order is deterministic,
        // unlike all_vars()' HashMap order).
        let named = enr.named_params();
        let mut norms = Vec::new();
        let mut sq = 0f64;
        for (_, t) in &named {
            let n = match grads.get(t) {
                Some(g) => g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64,
                None => 0.0,
            };
            norms.push(n.sqrt());
            sq += n;
        }
        // Sanity: no gradient outside the trainable params (fixed/PCA A is
        // intentionally excluded from both sides of the check).
        let mut total_check = 0f64;
        for var in enr.trainable_vars() {
            if let Some(g) = grads.get(&var) {
                total_check += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            }
        }
        let loss_v = out.ce_value;
        let per: Vec<String> = named
            .iter()
            .zip(&norms)
            .map(|((n, _), v)| format!("{n} {v:.3}"))
            .collect();
        println!(
            "  tau={tau:>2} kv_grad={kv_grad}: ce {loss_v:.4}, grad norm {:.4} ({}){}",
            sq.sqrt(),
            per.join(", "),
            if total_check - sq > 1e-6 {
                format!("  [WARN: foreign grads {:.3}]", (total_check - sq).sqrt())
            } else {
                String::new()
            }
        );
        rows.push((tau, kv_grad, out.ce_value, norms));
    }
    if rows.len() >= 2 {
        let (_, _, _, ref_norms) = &rows[0];
        for (tau, kv_grad, _, norms) in rows.iter().skip(1) {
            let ratios: Vec<String> = norms
                .iter()
                .zip(ref_norms)
                .map(|(n, r)| {
                    if r.abs() > 1e-30 {
                        format!("{:.3}", n / r)
                    } else {
                        "-".to_string()
                    }
                })
                .collect();
            println!(
                "  grad ratio (tau={tau} kv={kv_grad}) / (tau={} kv={}): [{}] (>= 0.99 per tensor => the cheaper config captures the gradient)",
                rows[0].0,
                rows[0].1,
                ratios.join(", ")
            );
        }
    }
    Ok(())
}

/// Peak-RSS diagnostic: current resident set in MiB (from /proc).
fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(|v| v.to_string()))
        .and_then(|v| v.parse::<f64>().ok())
        .map(|pages| pages * 4096.0 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

fn maybe_checkpoint(
    enr: &mut Enricher,
    last: &mut std::time::Instant,
    out_path: &str,
) -> Result<()> {
    if last.elapsed() >= std::time::Duration::from_secs(300) {
        enr.save(out_path)?;
        *last = std::time::Instant::now();
        println!("  checkpoint: {out_path}");
    }
    Ok(())
}

/// PCA calibration of the A matrix (fresh factored runs, a_mode = pca):
/// collect RMS-normalized states over the first training batches with a
/// no-grad cascade pass, then set A's rows to the top-`rank` eigenvectors of
/// the states' Gram matrix (candle has no symeig on CPU, so the top
/// eigenvectors are found by simultaneous orthogonal iteration — a few
/// matmuls per round, plenty accurate for a projection basis).
fn pca_calibrate(engine: &Engine, enr: &mut Enricher, batches: &[Batch]) -> Result<()> {
    let d = enr.spec().rank;
    let hidden = enr.hidden();
    println!(
        "train: PCA calibration of A (rank {d}) over the first {} batches...",
        PCA_BATCHES.min(batches.len())
    );
    enr.set_collect_shat(true);
    let mut rng = StdRng::seed_from_u64(4242);
    for batch in batches.iter().take(PCA_BATCHES.min(batches.len())) {
        engine.cascade_train_forward(
            enr,
            &batch.input_ids,
            &batch.targets,
            batch.answer_start,
            0, // no TBPTT window: pure forward, states collected
            false,
            1.0,
            1.0,
            &mut rng,
        )?;
    }
    enr.set_collect_shat(false);
    let (gram, n_rows) = enr
        .take_gram()
        .context("PCA calibration collected no states")?;
    println!("  collected {n_rows} state rows");

    // Power-style iteration: orthonormalize an initial [H, d] guess, then
    // repeatedly (gram, orthonormalize). Gram is symmetric PSD by
    // construction.
    let seed_a = crate::experiments::enrich::randn_rows(hidden, d, 1.0, 4243)?;
    let mut v = orthonormalize(&seed_a)?;
    for it in 0..PCA_ITERS {
        v = orthonormalize(&gram.matmul(&v)?)?;
        if it % 10 == 0 || it == PCA_ITERS - 1 {
            // Rayleigh-ish residual: relative change between iterations.
            let nv = orthonormalize(&gram.matmul(&v)?)?;
            let diff = nv.sub(&v)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
            println!("  pca iter {it}: subspace delta {:.3e}", diff.sqrt());
            v = nv;
        }
    }
    // A rows = top-d eigenvectors (columns of v), transposed to [d, H].
    let a = v.t()?.contiguous()?;
    enr.set_a(&a)?;
    Ok(())
}

const PCA_BATCHES: usize = 16;
const PCA_ITERS: usize = 60;

/// Gram-Schmidt orthonormalization of the columns of an [H, d] matrix.
fn orthonormalize(m: &Tensor) -> Result<Tensor> {
    let (h, d) = m.dims2()?;
    let mut cols: Vec<Tensor> = Vec::with_capacity(d);
    for j in 0..d {
        let mut c = m.narrow(1, j, 1)?.squeeze(1)?; // [H]
        for p in &cols {
            let dot = p.broadcast_mul(&c)?.sum_all()?.to_scalar::<f32>()?;
            c = c.sub(&p.affine(dot as f64, 0.0)?)?;
        }
        let norm = c.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-9);
        cols.push(c.affine(1.0 / norm as f64, 0.0)?);
    }
    let cols: Vec<Tensor> = cols
        .into_iter()
        .map(|c| c.reshape((h, 1)))
        .collect::<candle::Result<_>>()?;
    Ok(Tensor::cat(&cols, 1)?)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    engine: &mut Engine,
    tokenizer: &Tokenizer,
    chat_template: &str,
    device: &Device,
    warmup_steps: usize,
    st_steps: usize,
    out_path: &str,
    cfg: TrainConfig,
) -> Result<()> {
    println!("train: building dataset ({TRAIN_PAIRS} train / {HELDOUT_PAIRS} held-out pairs)...");
    let (mut train_examples, heldout) = build_dataset();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    shuffle(&mut train_examples, &mut rng);

    let fresh = match std::path::Path::new(out_path).exists() {
        true => {
            println!("train: resuming from existing checkpoint {out_path}");
            let loaded = Enricher::load(out_path)?;
            if loaded.spec() != &cfg.spec {
                anyhow::bail!(
                    "checkpoint variant {:?} != requested {:?} (delete {} to restart)",
                    loaded.spec(),
                    cfg.spec,
                    out_path
                );
            }
            loaded
        }
        false => {
            let e = Enricher::new(engine.hidden(), cfg.gate_bias, cfg.spec, cfg.seed)?;
            // Save the fresh (identity) checkpoint before any training so the
            // bit-exact baseline regression has a loadable artifact.
            e.save(out_path)?;
            println!(
                "train: fresh enricher saved to {out_path} ({} identity)",
                if cfg.spec.rank == 0 { "W=0" } else { "B=0" }
            );
            e
        }
    };
    let mut enr = fresh;
    let n_params: usize = enr.named_params().iter().map(|(_, t)| t.elem_count()).sum();
    let vars = enr.trainable_vars();
    let mut opt = candle_nn::AdamW::new_lr(vars.clone(), 1e-3)?;
    println!(
        "train: enricher {} (params {}, bias {:.2}, lr_warmup {}, floor {:.2}, gt {:.2}->{:.2}, tau {}, clip {:.2}, seed {})",
        cfg.spec.summary(),
        n_params,
        cfg.gate_bias,
        cfg.lr_warmup,
        cfg.gate_floor,
        cfg.gt_start,
        cfg.gt_end,
        cfg.tau,
        cfg.grad_clip,
        cfg.seed
    );

    let mut sample_rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));
    let batches = make_batches(&train_examples, tokenizer, chat_template, cfg.batch)?;
    let eval_batches = make_batches(&heldout, tokenizer, chat_template, 8)?;
    let eval_examples = heldout;
    println!(
        "train: {} batches (batch {}, accum {})",
        batches.len(),
        cfg.batch,
        cfg.accum
    );
    // PCA calibration of A (fresh factored runs only): one no-grad cascade
    // over the first batches collecting RMS-normalized states, then the top-d
    // eigenvectors of the Gram matrix become A's rows.
    if cfg.spec.rank > 0 && cfg.spec.a_mode == AMode::Pca {
        pca_calibrate(engine, &mut enr, &batches)?;
        enr.save(out_path)?;
        println!("train: PCA-calibrated checkpoint saved to {out_path}");
    }
    let total_batches = batches.len();
    let start = std::time::Instant::now();
    let mut last_checkpoint = start;
    let mut loss_sum = 0f64;
    let mut pen_sum = 0f64;
    let mut gnorm_sum = 0f64;
    let mut loss_n = 0usize;
    let mut step_n = 0usize;
    let mut low_gate_prints = 0usize;
    let mut acc: std::collections::HashMap<candle::TensorId, candle::Tensor> =
        std::collections::HashMap::new();
    let mut micro = 0usize;

    // TBPTT truncation + KV-path diagnostic before any training. Default
    // grid: the exact-gradient reference at tau=4 plus the training config.
    // Cheaper-first. (Do NOT auto-extend exact-KV configs here: tau+8 exact
    // peaked at 15 GB in-process. Use --train-sweep for explicit grids.)
    let grid: Vec<(usize, bool)> = if cfg.sweep.trim().is_empty() {
        let mut grid = vec![(4usize, true)];
        let c = (cfg.tau, cfg.kv_grad);
        if !grid.contains(&c) {
            grid.push(c);
        }
        grid
    } else {
        cfg.sweep
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                let (t, k) = s.split_once(':').unwrap_or((s, "1"));
                (
                    t.trim().parse::<usize>().unwrap_or(cfg.tau),
                    k.trim() != "0",
                )
            })
            .collect()
    };
    tau_sweep(engine, &enr, &batches[0], &grid, cfg.inj_l2)?;
    let _ = enr.take_gate_mean();
    let _ = enr.take_residual_stats();

    enr.set_gate_floor(cfg.gate_floor);
    println!(
        "train: CE warmup for {warmup_steps} steps (forced targets, gate floor {:.2})",
        cfg.gate_floor
    );
    for wi in 0..warmup_steps {
        let batch = &batches[wi % total_batches];
        let warm = if cfg.lr_warmup > 0 {
            (wi as f64 / cfg.lr_warmup as f64).min(1.0)
        } else {
            1.0
        };
        let lr = 1e-3 * warm;
        opt.set_learning_rate(lr);
        let (loss, pen, _ap, _aa, grads) = train_backward(
            engine,
            &enr,
            batch,
            cfg.tau,
            cfg.kv_grad,
            1.0,
            1.0,
            &mut sample_rng,
            cfg.inj_l2,
            cfg.accum,
        )?;
        accumulate_grads(&mut acc, &grads, &vars)?;
        micro += 1;
        let gnorm = if micro == cfg.accum {
            let g = accum_step(&mut acc, &vars, cfg.grad_clip, &mut opt)?;
            micro = 0;
            Some(g)
        } else {
            None
        };
        loss_sum += loss;
        pen_sum += pen;
        if let Some(g) = gnorm {
            gnorm_sum += g;
            step_n += 1;
        }
        loss_n += 1;
        if wi % 10 == 0 {
            let gm = enr.take_gate_mean().unwrap_or(f64::NAN);
            let um = enr.take_content_gate_mean().unwrap_or(f64::NAN);
            let (am, rm, dg) = enr.take_residual_stats().unwrap_or((f64::NAN, f64::NAN, 0));
            let dg_s = if dg > 0 {
                format!(" [degen {dg}]")
            } else {
                String::new()
            };
            println!(
                "  warmup {wi:>4}: loss {:.4} (lr {:.2e}, gate {:.3}, u {:.3}, alpha {am:.3}, rem {rm:.3}{dg_s}, pregate {:.3}, |g| {:.3}) [{:.0}s, rss {:.0} MiB]",
                loss_sum / loss_n.max(1) as f64,
                lr,
                gm,
                um,
                pen_sum / loss_n.max(1) as f64,
                gnorm_sum / step_n.max(1) as f64,
                start.elapsed().as_secs_f64(),
                rss_mib()
            );
            loss_sum = 0.0;
            pen_sum = 0.0;
            gnorm_sum = 0.0;
            loss_n = 0;
            step_n = 0;
            if gm.is_finite() && gm < 0.05 {
                low_gate_prints += 1;
            } else {
                low_gate_prints = 0;
            }
            if low_gate_prints >= 6 {
                let forensic = format!("{out_path}.collapsed");
                enr.save(&forensic)?;
                anyhow::bail!(
                    "gate collapsed (mean < 0.05 for 60 consecutive steps); state saved to {forensic}"
                );
            }
        }
        maybe_checkpoint(&mut enr, &mut last_checkpoint, out_path)?;
    }

    // Hard-sampled ST phase with scheduled forcing: p_forced annealed 1 -> 0,
    // sampling temperature gt annealed gt_start -> gt_end, gate floor annealed
    // 0.6 * warmup floor -> 0. CE stays on the targets throughout.
    println!("train: hard-sampled ST for {st_steps} steps");
    for si in 0..st_steps {
        let frac = si as f64 / st_steps.max(1) as f64;
        let forced_prob = 1.0 - frac;
        let gt = (cfg.gt_start - (cfg.gt_start - cfg.gt_end) * frac).max(cfg.gt_end);
        let floor = (cfg.gate_floor * 0.6) * (1.0 - frac);
        enr.set_gate_floor(floor);
        let batch = &batches[si % total_batches];
        let lr = 1e-4 + 5e-4 * (1.0 - 0.5 * frac);
        opt.set_learning_rate(lr);
        let (loss, pen, ap, aa, grads) = train_backward(
            engine,
            &enr,
            batch,
            cfg.tau,
            cfg.kv_grad,
            forced_prob,
            gt,
            &mut sample_rng,
            cfg.inj_l2,
            cfg.accum,
        )?;
        accumulate_grads(&mut acc, &grads, &vars)?;
        micro += 1;
        let gnorm = if micro == cfg.accum {
            let g = accum_step(&mut acc, &vars, cfg.grad_clip, &mut opt)?;
            micro = 0;
            Some(g)
        } else {
            None
        };
        loss_sum += loss;
        pen_sum += pen;
        if let Some(g) = gnorm {
            gnorm_sum += g;
            step_n += 1;
        }
        loss_n += 1;
        if si % 10 == 0 {
            let gm = enr.take_gate_mean().unwrap_or(f64::NAN);
            let um = enr.take_content_gate_mean().unwrap_or(f64::NAN);
            let (am, rm, dg) = enr.take_residual_stats().unwrap_or((f64::NAN, f64::NAN, 0));
            let dg_s = if dg > 0 {
                format!(" [degen {dg}]")
            } else {
                String::new()
            };
            println!(
                "  st {si:>4}: loss {:.4} (lr {:.2e}, gt {gt:.2}, forced {forced_prob:.2}, gate {:.3}, u {:.3}, alpha {am:.3}, rem {rm:.3}{dg_s}, pregate {:.3}, appl p/ans {:.3}/{:.3}, |g| {:.3}) [{:.0}s]",
                loss_sum / loss_n.max(1) as f64,
                lr,
                gm,
                um,
                pen_sum / loss_n.max(1) as f64,
                ap,
                aa,
                gnorm_sum / step_n.max(1) as f64,
                start.elapsed().as_secs_f64()
            );
            loss_sum = 0.0;
            pen_sum = 0.0;
            gnorm_sum = 0.0;
            loss_n = 0;
            step_n = 0;
            if gm.is_finite() && gm < 0.05 {
                low_gate_prints += 1;
            } else {
                low_gate_prints = 0;
            }
            if low_gate_prints >= 6 {
                let forensic = format!("{out_path}.collapsed");
                enr.save(&forensic)?;
                anyhow::bail!(
                    "gate collapsed (mean < 0.05 for 60 consecutive steps); state saved to {forensic}"
                );
            }
        }
        maybe_checkpoint(&mut enr, &mut last_checkpoint, out_path)?;
    }
    enr.set_gate_floor(0.0);

    if cfg.skip_eval {
        enr.save(out_path)?;
        println!("train: saved enricher to {out_path} (eval skipped)");
        return Ok(());
    }

    println!("train: held-out evaluation");
    let baseline = Enricher::new(engine.hidden(), 0.0, EnricherSpec::default(), 0)?;
    let base_tf = teacher_forced_em(engine, &eval_batches, device, &baseline)?;
    let trained_tf = teacher_forced_em(engine, &eval_batches, device, &enr)?;
    println!(
        "train: teacher-forced EM (transition-paired): baseline {:.1}% -> trained {:.1}%",
        100.0 * base_tf,
        100.0 * trained_tf
    );
    enr.save(out_path)?;
    println!("train: saved enricher to {out_path}");

    engine.attach_enricher(baseline)?;
    let base_fr = free_running_eval(
        engine,
        tokenizer,
        chat_template,
        &eval_examples,
        device,
        false,
    )?;
    let base_fr_guided = free_running_eval(
        engine,
        tokenizer,
        chat_template,
        &eval_examples,
        device,
        true,
    )?;
    engine.attach_enricher(enr)?;
    let trained_fr = free_running_eval(
        engine,
        tokenizer,
        chat_template,
        &eval_examples,
        device,
        false,
    )?;
    let trained_fr_guided = free_running_eval(
        engine,
        tokenizer,
        chat_template,
        &eval_examples,
        device,
        true,
    )?;
    println!(
        "train: free-running EM: baseline {:.1}% -> trained {:.1}% (first-token-guided: baseline {:.1}% -> trained {:.1}%)",
        100.0 * base_fr,
        100.0 * trained_fr,
        100.0 * base_fr_guided,
        100.0 * trained_fr_guided
    );
    Ok(())
}

/// Teacher-forced exact match (transition-paired): cascade forward with
/// forced targets and no graph; argmax over the transition logits vs the
/// answer targets. `enr` = a fresh zero enricher gives the exact baseline.
fn teacher_forced_em(
    engine: &Engine,
    batches: &[Batch],
    _device: &Device,
    enr: &Enricher,
) -> Result<f64> {
    let mut rng = StdRng::seed_from_u64(1);
    let mut correct = 0usize;
    let mut total = 0usize;
    for batch in batches {
        let out = engine.cascade_train_forward(
            enr,
            &batch.input_ids,
            &batch.targets,
            batch.answer_start,
            0,
            false,
            1.0,
            1.0,
            &mut rng,
        )?;
        let preds = out
            .logits
            .to_dtype(DType::F32)?
            .argmax(2)?
            .to_vec2::<u32>()?;
        let targets = batch.targets.to_vec2::<i64>()?;
        for (row, tgt_row) in targets.iter().enumerate() {
            for (slot, &t) in tgt_row.iter().enumerate() {
                if t < 0 {
                    continue;
                }
                total += 1;
                if preds[row][slot] as i64 == t {
                    correct += 1;
                }
            }
        }
    }
    Ok(correct as f64 / total.max(1) as f64)
}

/// Free-running greedy decode over held-out examples with an enricher
/// attached (the cascade prefill + decode path; a fresh zero enricher is the
/// exact baseline). With `guided_first` the first answer token is
/// teacher-forced from the target (separates first-step bootstrap failures
/// from free-running destabilization).
fn free_running_eval(
    engine: &mut Engine,
    tokenizer: &Tokenizer,
    chat_template: &str,
    examples: &[Example],
    _device: &Device,
    guided_first: bool,
) -> Result<f64> {
    let end_of_turn = tokenizer
        .token_to_id("<end_of_turn>")
        .context("no <end_of_turn>")?;
    let mut correct = 0usize;
    let mut shown = 0usize;
    for ex in examples {
        let rendered = chat::render(chat_template, None, &ex.prompt)?;
        let prompt_tokens = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let forced_first = if guided_first {
            Some(
                tokenizer
                    .encode(ex.answer.as_str(), false)
                    .map_err(E::msg)?
                    .get_ids()[0],
            )
        } else {
            None
        };
        engine.clear_kv_cache();
        let mut logits = None;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = Some(engine.cascade_step(tok, pos)?);
        }
        let mut logits = logits.context("empty prompt")?;
        let mut text = String::new();
        for (pos, step) in (prompt_tokens.len()..).zip(0..8) {
            let lg = logits
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let argmax = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            let best = if step == 0 {
                forced_first.unwrap_or(argmax)
            } else {
                argmax
            };
            if best == end_of_turn {
                break;
            }
            text.push_str(&tokenizer.decode(&[best], true).unwrap_or_default());
            logits = engine.cascade_step(best, pos)?;
        }
        if score(ex.task, &ex.answer, &text) {
            correct += 1;
        } else if shown < 5 {
            println!(
                "    miss [{}]: {:?} -> {:?}",
                ex.task,
                ex.answer,
                text.trim()
            );
            shown += 1;
        }
    }
    Ok(correct as f64 / examples.len() as f64)
}

/// Probe a trained enricher checkpoint: cascade over held-out batches
/// (forced tokens, no graph) and report the gate / injection statistics.
/// (The M1 soft-mixture probe metrics are obsolete: v2 feeds hard tokens.)
pub fn run_probe(
    engine: &Engine,
    tokenizer: &Tokenizer,
    chat_template: &str,
    _device: &Device,
    checkpoint: &str,
) -> Result<()> {
    let heldout = build_dataset().1;
    let batches = make_batches(&heldout, tokenizer, chat_template, 8)?;
    let enr = Enricher::load(checkpoint)?;
    let mut rng = StdRng::seed_from_u64(999);
    let n_batches = batches.len().min(5);
    println!("probe: checkpoint {checkpoint}, {n_batches} held-out batches");
    let mut preg_sum = 0f64;
    let mut appl_p_sum = 0f64;
    let mut appl_a_sum = 0f64;
    let mut ce_sum = 0f64;
    for batch in batches.iter().take(n_batches) {
        let out = engine.cascade_train_forward(
            &enr,
            &batch.input_ids,
            &batch.targets,
            batch.answer_start,
            0,
            false,
            1.0,
            1.0,
            &mut rng,
        )?;
        preg_sum += out.pen.to_scalar::<f32>()? as f64;
        appl_p_sum += out.applied_prompt;
        appl_a_sum += out.applied_answer;
        ce_sum += out.ce_value;
    }
    let n = n_batches as f64;
    let gate = enr.take_gate_mean();
    let (am, rm, dg) = enr.take_residual_stats().unwrap_or((f64::NAN, f64::NAN, 0));
    println!(
        "probe: gate {:?} | pregate rel {:.3} | applied rel prompt {:.3} / answer {:.3} | alpha {am:.3} rem {rm:.3} degen {dg} | ce {:.3}",
        gate,
        preg_sum / n,
        appl_p_sum / n,
        appl_a_sum / n,
        ce_sum / n
    );
    Ok(())
}
