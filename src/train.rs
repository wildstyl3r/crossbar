use anyhow::{Context, Error as E, Result};
use candle::Result as CResult;
use candle::{DType, Device, Tensor};
use candle_nn::Optimizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;

use crate::chat;
use crate::enrich::Enricher;
use crate::Engine;

const R: usize = 192;
const K: usize = 8;
const TRAIN_PAIRS: usize = 2000;
const HELDOUT_PAIRS: usize = 100;

/// Training hyperparameters. Defaults restore the run-2 recipe (the only
/// one that ever converged): constant warmup lr, no gate floor, gate bias
/// 0, Gumbel temperature annealed 1.0 -> 0.3.
pub struct TrainConfig {
    /// Initial gate bias of a fresh enricher (0 => sigmoid 0.5; run 4 used
    /// +2.0).
    pub gate_bias: f64,
    /// Linear lr ramp length at the start of warmup (0 = constant; run 4
    /// used 50).
    pub lr_warmup: usize,
    /// Gate floor during warmup (0 = none; run 4 used 0.5). The ST phase
    /// anneals a floor of 0.6 * gate_floor -> 0.
    pub gate_floor: f64,
    /// Starting Gumbel temperature of the ST phase, annealed linearly to
    /// `gt_end` (run 2: 1.0 over 300 steps; post-run-2 code: 0.5).
    pub gt_start: f64,
    /// Anneal target of the ST sampling temperature (0.1 ~= the greedy
    /// deployment regime).
    pub gt_end: f64,
    /// Relative injection penalty: `loss + inj_l2 * mean(
    /// ||g * W_o(pooled)||^2 / ||e*sqrt(h)||^2 )`. The probe on the run-5
    /// checkpoint showed the unpenalized injection reaching ~131x the input
    /// norm (the module replaced the token identity with pooled states);
    /// useful hand-set enrichments live at O(0.25-1)x.
    pub inj_l2: f64,
    /// Seed for the training-data shuffle and the ST sampling rng.
    pub seed: u64,
    /// Skip the terminal evaluation (smoke runs).
    pub skip_eval: bool,
}

pub struct Example {
    pub task: &'static str,
    pub prompt: String,
    pub answer: String,
}

const AGENTS: &[&str] = &["zorp", "kit", "milu"];
const OBJECTS: &[&str] = &["blicket", "murble", "wug", "toma"];

fn shuffle<T>(v: &mut [T], rng: &mut StdRng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
}

/// Chain-of-custody state tracking: statements move objects between holders;
/// the question asks who has one object. Depth-2/3 chains for the target
/// object, one distractor statement about a second object.
fn build_state_track(rng: &mut StdRng) -> Example {
    let mut holders: Vec<&str> = AGENTS.to_vec();
    shuffle(&mut holders, rng);
    let mut objects: Vec<&str> = OBJECTS.to_vec();
    shuffle(&mut objects, rng);
    let target_obj = objects[0];
    let distractor_obj = objects[1];
    let depth = rng.random_range(2..=3usize);

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("The {target_obj} is with {}.", holders[0]));
    lines.push(format!("The {distractor_obj} is with {}.", holders[2]));
    let mut cur = holders[0];
    for step in 0..depth {
        let next = holders[1 + (step % 2)];
        lines.push(format!("{cur} gives the {target_obj} to {next}."));
        cur = next;
    }
    lines.push(format!("{} takes the {distractor_obj}.", holders[2]));
    let answer = {
        let mut c = cur.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }
    };
    shuffle(&mut lines, rng);
    Example {
        task: "state-track",
        prompt: format!(
            "{}\nQuestion: who has the {target_obj}?\nAnswer with only the name.",
            lines.join(" ")
        ),
        answer,
    }
}

const NONSENSE: &[&str] = &[
    "dax", "wug", "zorpie", "toma", "rugg", "fep", "kiki", "gorb", "pil", "nad", "mell", "lorp",
    "seng", "tob", "cootie",
];

/// Induction with distractors: 6 relevant word->number statements plus 2-3
/// distractor statements; the 6th relevant pair is queried.
fn build_induce_d(rng: &mut StdRng) -> Example {
    let mut pool = NONSENSE.to_vec();
    shuffle(&mut pool, rng);
    let chosen: Vec<&str> = pool[..6].to_vec();
    let distractors: Vec<&str> = pool[6..9].to_vec();
    let numbers: Vec<i64> = (0..6).map(|_| rng.random_range(1..100)).collect();
    let mut stmts: Vec<String> = chosen
        .iter()
        .zip(&numbers)
        .map(|(w, n)| format!("{w} is {n}"))
        .collect();
    let n_distract = rng.random_range(2..=3usize);
    for d in distractors.iter().take(n_distract) {
        stmts.push(format!("{d} is red."));
    }
    shuffle(&mut stmts, rng);
    let query = chosen[5];
    let answer = numbers[5].to_string();
    Example {
        task: "induce-d",
        prompt: format!(
            "{}.\nQuestion: {query} is?\nAnswer with only the number.",
            stmts.join(". ")
        ),
        answer,
    }
}

fn build_dataset() -> (Vec<Example>, Vec<Example>) {
    let mut rng = StdRng::seed_from_u64(777);
    let mut train = Vec::new();
    for _ in 0..TRAIN_PAIRS {
        train.push(build_state_track(&mut rng));
        train.push(build_induce_d(&mut rng));
    }
    let mut heldout = Vec::new();
    for _ in 0..HELDOUT_PAIRS {
        heldout.push(build_state_track(&mut rng));
        heldout.push(build_induce_d(&mut rng));
    }
    (train, heldout)
}

fn first_nonempty_line(text: &str) -> &str {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(str::trim)
        .unwrap_or("")
}

fn last_int(s: &str) -> Option<i64> {
    let mut last = None;
    let mut cur: Option<i64> = None;
    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            cur = Some(cur.unwrap_or(0).saturating_mul(10).saturating_add(d as i64));
        } else if let Some(v) = cur.take() {
            last = Some(v);
        }
    }
    if let Some(v) = cur {
        last = Some(v);
    }
    last
}

/// Exact-match scoring for the two task families.
fn score(task: &str, answer: &str, text: &str) -> bool {
    let line = first_nonempty_line(text).to_lowercase();
    match task {
        "state-track" => {
            let words: Vec<String> = line
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { ' ' })
                .collect::<String>()
                .split_whitespace()
                .map(str::to_string)
                .collect();
            let answer_lc = answer.to_lowercase();
            words.iter().any(|w| *w == answer_lc)
        }
        _ => last_int(&line) == answer.parse::<i64>().ok(),
    }
}

fn encode_answer(tokenizer: &Tokenizer, ex: &Example) -> Result<Vec<u32>> {
    let text = format!("{} <end_of_turn>", ex.answer);
    let ids = tokenizer.encode(text.as_str(), false).map_err(E::msg)?.get_ids().to_vec();
    Ok(ids)
}

struct Batch {
    /// [b, seq] token ids (prompt tokens, then answer tokens, then pads).
    input_ids: Tensor,
    /// Real (unpadded) length per row.
    lens: Vec<usize>,
    /// Answer slots occupy [answer_start, answer_start + answer_len).
    answer_start: usize,
    answer_len: usize,
    /// Target ids [b, answer_len], -100 on pads.
    targets: Tensor,
    /// Valid answer slots per row.
    valid: Vec<usize>,
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
        let mut lens = Vec::with_capacity(b);
        let mut valid = Vec::with_capacity(b);
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
            lens.push(p.len() + a.len());
            valid.push(a.len());
        }
        batches.push(Batch {
            input_ids: Tensor::from_vec(flat, (b, seq_len), &Device::Cpu)?,
            lens,
            answer_start: prompt_len,
            answer_len,
            targets: Tensor::from_vec(targets, (b, answer_len), &Device::Cpu)?,
            valid,
        });
        i = end;
    }
    Ok(batches)
}

fn gumbel_softmax(
    logits: &[f32],
    temperature: f64,
    rng: &mut StdRng,
    device: &Device,
) -> CResult<Tensor> {
    use rand::Rng as _;
    let g: Vec<f32> = logits
        .iter()
        .map(|&l| {
            let u: f32 = rng.random_range(1e-9..1.0);
            ((l as f64 + (-(u as f64).ln()).ln()) / temperature) as f32
        })
        .collect();
    let max = g.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = g.iter().map(|&g| (g - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    Tensor::from_vec(probs, logits.len(), device)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// Probe a trained enricher checkpoint: quantify the ST soft-mixture
/// artifact (did the module learn to denoise Gumbel mixtures toward the
/// clean token embedding?) and the deployment-relevant distortion of clean
/// inputs. Prints mean statistics over answer slots of a few held-out
/// batches:
/// - c_in:  cos(e_mix, e_clean) — input smear at the given gumbel temp;
/// - c_st:  cos(enrich(e_mix), e_clean) — st path output vs clean (the
///          denoising signal is c_st > c_in);
/// - c_dep: cos(enrich(e_clean), e_clean) — deployment case (want ~1.0);
/// - inj:   ||enrich(e_clean) - e_clean*sqrt(h)|| / ||e_clean*sqrt(h)||;
/// - scale: ||e_mix_scaled|| / ||e_clean_raw|| (the double-sqrt(h) factor
///          shows up as ~sqrt(hidden) for peaked mixtures).
pub fn run_probe(
    engine: &Engine,
    tokenizer: &Tokenizer,
    chat_template: &str,
    device: &Device,
    checkpoint: &str,
) -> Result<()> {
    let heldout = build_dataset().1;
    let batches = make_batches(&heldout, tokenizer, chat_template, 8)?;
    let enr = Enricher::load(checkpoint, engine.hidden(), R, K)?;
    let mut sample_rng = StdRng::seed_from_u64(999);
    let n_batches = batches.len().min(5);
    println!(
        "probe: checkpoint {checkpoint}, {n_batches} held-out batches, r={R} k={K}"
    );
    for &gt in &[1.0f64, 0.3] {
        let mut c_in_sum = 0f64;
        let mut c_st_sum = 0f64;
        let mut c_dep_sum = 0f64;
        let mut inj_sum = 0f64;
        let mut scale_sum = 0f64;
        let mut gate_st_sum = 0f64;
        let mut gate_dep_sum = 0f64;
        let mut n = 0usize;
        for batch in batches.iter().take(n_batches) {
            let (states, pass_a_logits) = engine.forward_inputs_train(
                &engine.embed_scale(&batch.input_ids, &batch.lens)?,
                &batch.lens,
                batch.answer_start,
                batch.answer_len,
            )?;
            let logits_v = pass_a_logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            for row in 0..batch.lens.len() {
                for slot in 0..batch.valid[row] {
                    let pos = batch.answer_start + slot;
                    let mut window: Vec<Tensor> = Vec::new();
                    for j in pos.saturating_sub(K)..pos {
                        if j < batch.lens[row] {
                            let st = states
                                .narrow(0, row, 1)?
                                .narrow(1, j, 1)?
                                .squeeze(1)?
                                .squeeze(0)?
                                .contiguous()?;
                            window.push(st);
                        }
                    }
                    let probs = gumbel_softmax(&logits_v[row][slot], gt, &mut sample_rng, device)?;
                    let idx = probs.argmax(0)?.to_scalar::<u32>()?;
                    let tok = Tensor::new(&[idx], &Device::Cpu)?;
                    let e_clean_raw = engine.embed_one(&tok)?; // [hidden], unscaled
                    let e_mix = engine
                        .expected_from_probs(&probs.unsqueeze(0)?)? // scaled by sqrt(h)
                        .squeeze(0)?
                        .to_vec1::<f32>()?;
                    let e_clean_v = e_clean_raw.to_vec1::<f32>()?;

                    // ST path exactly as trained: pre-scaled mixture in.
                    let (y_st, _pen_st) = enr.enrich(
                        &Tensor::from_vec(e_mix.clone(), e_mix.len(), device)?,
                        &window,
                    )?;
                    let y_st = y_st.to_vec1::<f32>()?;
                    let gate_st = enr.take_gate_mean().unwrap_or(f64::NAN);
                    // Deployment path: raw clean embedding in.
                    let (y_dep, _pen_dep) = enr.enrich(&e_clean_raw, &window)?;
                    let y_dep = y_dep.to_vec1::<f32>()?;
                    let gate_dep = enr.take_gate_mean().unwrap_or(f64::NAN);

                    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                    let e_norm = norm(&e_clean_v);
                    let c_in = cosine(&e_mix, &e_clean_v);
                    let c_st = cosine(&y_st, &e_clean_v);
                    let c_dep = cosine(&y_dep, &e_clean_v);
                    let e_scaled_norm = e_norm * (engine.hidden() as f64).sqrt();
                    let inj = {
                        let d: Vec<f32> = y_dep
                            .iter()
                            .zip(&e_clean_v)
                            .map(|(y, e)| y - (*e as f32) * (engine.hidden() as f32).sqrt())
                            .collect();
                        norm(&d) / e_scaled_norm.max(1e-9)
                    };
                    let scale = norm(&e_mix) / e_norm.max(1e-9);

                    c_in_sum += c_in;
                    c_st_sum += c_st;
                    c_dep_sum += c_dep;
                    inj_sum += inj;
                    scale_sum += scale;
                    if gate_st.is_finite() {
                        gate_st_sum += gate_st;
                    }
                    if gate_dep.is_finite() {
                        gate_dep_sum += gate_dep;
                    }
                    n += 1;
                }
            }
        }
        let n = n.max(1) as f64;
        let (c_in, c_st, c_dep, inj, scale) = (
            c_in_sum / n,
            c_st_sum / n,
            c_dep_sum / n,
            inj_sum / n,
            scale_sum / n,
        );
        let (gate_st, gate_dep) = (gate_st_sum / n, gate_dep_sum / n);
        println!(
            "probe gt={gt:.2}: c_in {c_in:.3} -> c_st {c_st:.3} (denoise {:+.3}) | c_dep {c_dep:.3} | inj {inj:.3} | scale {scale:.2} | gate st {gate_st:.3} dep {gate_dep:.3} (n={:.0})",
            c_st - c_in,
            n
        );
    }
    Ok(())
}

/// Build the enriched answer-slot input embeddings for a batch from
/// Pass-A states. `gumbel` (pass_a_logits [b, A, V], temperature) replaces
/// the teacher-forced token embedding with the exact embedding of one
/// *sampled* token (hard forward; the sampled index comes from the
/// Gumbel-softmax of the Pass-A logits at temperature `gt`) — matching the
/// deployment input distribution. Returns the batch inputs and the mean
/// relative injection energy `||g * W_o(pooled)||^2 / ||e * sqrt(h)||^2`
/// over the batch's answer slots (for the training-time penalty).
fn enriched_inputs(
    engine: &Engine,
    enr: &Enricher,
    batch: &Batch,
    pass_a_states: &Tensor,
    gumbel: Option<(&Tensor, f64)>,
    device: &Device,
    rng: &mut StdRng,
) -> Result<(Tensor, Tensor)> {
    let b = batch.lens.len();
    let (_, seq_len) = batch.input_ids.dims2()?;
    let base = engine.embed_scale(&batch.input_ids, &batch.lens)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(b);
    let mut pens: Vec<Tensor> = Vec::new();
    for row in 0..b {
        let mut row_base = base.narrow(0, row, 1)?.squeeze(0)?; // [seq, hidden]
        for slot in 0..batch.valid[row] {
            let pos = batch.answer_start + slot;
            let mut window = Vec::new();
            for j in pos.saturating_sub(K)..pos {
                if j < batch.lens[row] {
                    let st = pass_a_states
                        .narrow(0, row, 1)?
                        .narrow(1, j, 1)?
                        .squeeze(1)?
                        .squeeze(0)?
                        .contiguous()?;
                    window.push(st);
                }
            }
            let e = match &gumbel {
                Some((logits, gt)) => {
                    let lg = logits
                        .narrow(0, row, 1)?
                        .narrow(1, slot, 1)?
                        .squeeze(0)?
                        .squeeze(0)?
                        .to_vec1::<f32>()?;
                    let probs = gumbel_softmax(&lg, *gt, rng, device)?;
                    let idx = probs.argmax(0)?.to_scalar::<u32>()?;
                    engine.embed_one(&Tensor::new(&[idx], device)?)?
                }
                None => {
                    let tok = batch
                        .input_ids
                        .narrow(0, row, 1)?
                        .narrow(1, pos, 1)?
                        .squeeze(0)?
                        .squeeze(0)?;
                    engine.embed_one(&tok)?
                }
            };
            let (enriched, pen) = enr.enrich(&e, &window)?; // [hidden], scalar
            pens.push(pen);
            let left = row_base.narrow(0, 0, pos)?;
            let right = row_base.narrow(0, pos + 1, seq_len - pos - 1)?;
            let mid = enriched.unsqueeze(0)?;
            row_base = Tensor::cat(&[&left, &mid, &right], 0)?;
        }
        rows.push(row_base);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    let inputs = Tensor::stack(&refs, 0)?;
    let pen = if pens.is_empty() {
        Tensor::new(&[0.0f32], device)?
    } else {
        Tensor::stack(&pens, 0)?.mean_all()?
    };
    Ok((inputs, pen))
}

fn masked_ce(logits: &Tensor, targets: &Tensor, device: &Device) -> Result<Tensor> {
    let (b, a, vocab) = logits.dims3()?;
    let lg = logits.to_dtype(DType::F32)?.reshape((b * a, vocab))?;
    let tg = targets.flatten_all()?.to_vec1::<i64>()?;
    if !tg.iter().any(|&t| t >= 0) {
        anyhow::bail!("no valid targets in batch");
    }
    let log_probs = candle_nn::ops::log_softmax(&lg, 1)?;
    let idx = Tensor::from_vec(
        tg.iter().map(|&t| if t >= 0 { t as u32 } else { 0u32 }).collect(),
        (b * a,),
        device,
    )?;
    let picked = log_probs.gather(&idx.unsqueeze(1)?, 1)?.reshape((b * a,))?;
    let per = picked.reshape((b * a, 1))?.affine(-1.0, 0.0)?;
    let mask_t = Tensor::from_vec(
        tg.iter().map(|&t| if t >= 0 { 1.0f32 } else { 0.0 }).collect(),
        (b * a, 1),
        device,
    )?;
    let n_valid = tg.iter().filter(|&&t| t >= 0).count() as f64;
    let loss = (&per * mask_t)?.sum_all()?.affine(1.0 / n_valid, 0.0)?;
    Ok(loss)
}

fn maybe_checkpoint(enr: &mut Enricher, last: &mut std::time::Instant, out_path: &str) -> Result<()> {
    if last.elapsed() >= std::time::Duration::from_secs(300) {
        enr.save(out_path)?;
        *last = std::time::Instant::now();
        println!("  checkpoint: {out_path}");
    }
    Ok(())
}

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

    let mut enr = match std::path::Path::new(out_path).exists() {
        true => {
            println!("train: resuming from existing checkpoint {out_path}");
            Enricher::load(out_path, engine.hidden(), R, K)?
        }
        false => Enricher::new(engine.hidden(), R, K, cfg.gate_bias, &Device::Cpu)?,
    };
    let vars = enr.vars();
    let mut opt = candle_nn::AdamW::new_lr(vars, 1e-3)?;
    println!(
        "train: enricher r={R} k={K} (bias {:.2}, lr_warmup {}, floor {:.2}, gt_start {:.2}, seed {})",
        cfg.gate_bias, cfg.lr_warmup, cfg.gate_floor, cfg.gt_start, cfg.seed
    );

    let mut sample_rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));
    let batches = make_batches(&train_examples, tokenizer, chat_template, 8)?;
    let eval_batches = make_batches(&heldout, tokenizer, chat_template, 8)?;
    let eval_examples = heldout;
    println!("train: {} batches", batches.len());
    let total_batches = batches.len();
    let start = std::time::Instant::now();
    let mut last_checkpoint = start;
    let mut loss_sum = 0f64;
    let mut pen_sum = 0f64;
    let mut loss_n = 0usize;
    let mut low_gate_prints = 0usize;

    enr.set_gate_floor(cfg.gate_floor);
    println!(
        "train: CE warmup for {warmup_steps} steps (gate floor {:.2})",
        cfg.gate_floor
    );
    for wi in 0..warmup_steps {
        let batch = &batches[wi % total_batches];
        let decay = 1.0 - 0.5 * (wi as f64 / warmup_steps.max(1) as f64);
        let warm = if cfg.lr_warmup > 0 {
            (wi as f64 / cfg.lr_warmup as f64).min(1.0)
        } else {
            1.0
        };
        let lr = 1e-3 * decay * warm;
        opt.set_learning_rate(lr);
        let (states, _) = engine.forward_inputs_train(
            &engine.embed_scale(&batch.input_ids, &batch.lens)?,
            &batch.lens,
            batch.answer_start,
            batch.answer_len,
        )?;
        let (inputs, pen) =
            enriched_inputs(engine, &enr, batch, &states, None, device, &mut sample_rng)?;
        let (_, pass_b_logits) =
            engine.forward_inputs_train(&inputs, &batch.lens, batch.answer_start, batch.answer_len)?;
        let mut loss = masked_ce(&pass_b_logits, &batch.targets, device)?;
        let pen_v = pen.to_scalar::<f32>()? as f64;
        if cfg.inj_l2 > 0.0 {
            loss = (loss + (pen * cfg.inj_l2)?)?;
        }
        opt.backward_step(&loss)?;
        loss_sum += loss.to_scalar::<f32>()? as f64;
        pen_sum += pen_v;
        loss_n += 1;
        if wi % 10 == 0 {
            let gm = enr.take_gate_mean().unwrap_or(f64::NAN);
            println!(
                "  warmup {wi:>4}: loss {:.4} (lr {:.2e}, gate {:.3}, inj_l2 {:.3}) [{:.0}s]",
                loss_sum / loss_n.max(1) as f64,
                lr,
                gm,
                pen_sum / loss_n.max(1) as f64,
                start.elapsed().as_secs_f64()
            );
            loss_sum = 0.0;
            pen_sum = 0.0;
            loss_n = 0;
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

    // Gentler ST transition: keep an annealed gate floor (0.6 * warmup
    // floor -> 0) and an annealed Gumbel temperature; the teacher-forced ->
    // soft-mixture shift was shown to re-trigger gate collapse (run 4,
    // st 0-10).
    println!("train: guided ST-Gumbel for {st_steps} steps");
    for si in 0..st_steps {
        let floor = (cfg.gate_floor * 0.6) * (1.0 - si as f64 / st_steps.max(1) as f64);
        enr.set_gate_floor(floor);
        let gt =
            (cfg.gt_start - (cfg.gt_start - cfg.gt_end) * si as f64 / st_steps.max(1) as f64)
                .max(cfg.gt_end);
        let batch = &batches[si % total_batches];
        let lr = 1e-4 + 5e-4 * (1.0 - 0.5 * (si as f64 / st_steps.max(1) as f64));
        opt.set_learning_rate(lr);
        let (states, pass_a_logits) = engine.forward_inputs_train(
            &engine.embed_scale(&batch.input_ids, &batch.lens)?,
            &batch.lens,
            batch.answer_start,
            batch.answer_len,
        )?;
        let (inputs, pen) = enriched_inputs(
            engine,
            &enr,
            batch,
            &states,
            Some((&pass_a_logits, gt)),
            device,
            &mut sample_rng,
        )?;
        let (_, pass_b_logits) =
            engine.forward_inputs_train(&inputs, &batch.lens, batch.answer_start, batch.answer_len)?;
        let mut loss = masked_ce(&pass_b_logits, &batch.targets, device)?;
        let pen_v = pen.to_scalar::<f32>()? as f64;
        if cfg.inj_l2 > 0.0 {
            loss = (loss + (pen * cfg.inj_l2)?)?;
        }
        opt.backward_step(&loss)?;
        loss_sum += loss.to_scalar::<f32>()? as f64;
        pen_sum += pen_v;
        loss_n += 1;
        if si % 10 == 0 {
            let gm = enr.take_gate_mean().unwrap_or(f64::NAN);
            println!(
                "  st {si:>4}: loss {:.4} (lr {:.2e}, gt {gt:.2}, gate {:.3}, inj_l2 {:.3}) [{:.0}s]",
                loss_sum / loss_n.max(1) as f64,
                lr,
                gm,
                pen_sum / loss_n.max(1) as f64,
                start.elapsed().as_secs_f64()
            );
            loss_sum = 0.0;
            pen_sum = 0.0;
            loss_n = 0;
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

    if cfg.skip_eval {
        enr.save(out_path)?;
        println!("train: saved enricher to {out_path} (eval skipped)");
        return Ok(());
    }

    println!("train: held-out evaluation");
    let base_tf = teacher_forced_em(engine, &eval_batches, device, None)?;
    let trained_tf = teacher_forced_em(engine, &eval_batches, device, Some(&enr))?;
    println!(
        "train: teacher-forced EM: baseline {:.1}% -> trained {:.1}%",
        100.0 * base_tf,
        100.0 * trained_tf
    );
    enr.save(out_path)?;
    println!("train: saved enricher to {out_path}");

    let fresh = Enricher::new(engine.hidden(), R, K, 0.0, &Device::Cpu)?;
    engine.attach_enricher(fresh)?;
    let base_fr = free_running_eval(engine, tokenizer, chat_template, &eval_examples, device, false)?;
    let base_fr_guided =
        free_running_eval(engine, tokenizer, chat_template, &eval_examples, device, true)?;
    engine.attach_enricher(enr)?;
    let trained_fr = free_running_eval(engine, tokenizer, chat_template, &eval_examples, device, false)?;
    let trained_fr_guided =
        free_running_eval(engine, tokenizer, chat_template, &eval_examples, device, true)?;
    println!(
        "train: free-running EM: baseline {:.1}% -> trained {:.1}% (first-token-guided: baseline {:.1}% -> trained {:.1}%)",
        100.0 * base_fr,
        100.0 * trained_fr,
        100.0 * base_fr_guided,
        100.0 * trained_fr_guided
    );
    Ok(())
}

/// Teacher-forced exact match: argmax over the logits of the valid answer
/// slots. With `enr = None` the Pass-A (baseline) logits are used; with
/// `Some(enr)` the Pass-B logits with enriched answer inputs.
fn teacher_forced_em(
    engine: &Engine,
    batches: &[Batch],
    device: &Device,
    enr: Option<&Enricher>,
) -> Result<f64> {
    let mut correct = 0usize;
    let mut total = 0usize;
    for batch in batches {
        let (states, pass_a_logits) = engine.forward_inputs_train(
            &engine.embed_scale(&batch.input_ids, &batch.lens)?,
            &batch.lens,
            batch.answer_start,
            batch.answer_len,
        )?;
        let logits = match enr {
            None => pass_a_logits,
            Some(enr) => {
                let mut rng = StdRng::seed_from_u64(1);
                let (inputs, _) =
                    enriched_inputs(engine, enr, batch, &states, None, device, &mut rng)?;
                let (_, l) = engine.forward_inputs_train(
                    &inputs,
                    &batch.lens,
                    batch.answer_start,
                    batch.answer_len,
                )?;
                l
            }
        };
        let targets = batch.targets.to_vec2::<i64>()?;
        let preds = logits
            .to_dtype(DType::F32)?
            .argmax(2)?
            .to_vec2::<u32>()?;
        for (row, valid) in batch.valid.iter().enumerate() {
            for slot in 0..*valid {
                total += 1;
                if preds[row][slot] as i64 == targets[row][slot] {
                    correct += 1;
                }
            }
        }
    }
    Ok(correct as f64 / total.max(1) as f64)
}

/// Free-running greedy decode over held-out examples with the enricher
/// attached (states buffer maintained by the model itself). With
/// `guided_first` the first answer token is teacher-forced from the target
/// (a diagnostic that separates first-step bootstrap failures from
/// free-running destabilization — run 2's loop attractors).
fn free_running_eval(
    engine: &mut Engine,
    tokenizer: &Tokenizer,
    chat_template: &str,
    examples: &[Example],
    device: &Device,
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
        let input = Tensor::from_vec(prompt_tokens.clone(), (1, prompt_tokens.len()), device)?;
        let mut logits = engine.forward(&input, 0)?;
        let mut text = String::new();
        let mut pos = prompt_tokens.len();
        for step in 0..8 {
            let lg = logits
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let argmax = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
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
            let (l, _) = engine.forward_feedback(&logits, best, pos)?;
            logits = l;
            pos += 1;
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
