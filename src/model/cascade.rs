//! Cascade training forward (TBPTT) for the F32 backend.

use candle::{DType, Module, Result, Tensor};
use rand::Rng as _;

use crate::enrich::Enricher;

use super::Model;
use super::layers::{DecoderLayer, repeat_kv};

/// Output of the cascade training forward.
pub struct CascadeTrainOut {
    /// Graph loss for the backward pass: `sum(normed_t * g_t)` over the
    /// in-graph transition slots, whose gradient w.r.t. the states is
    /// exactly d(CE)/d(state) (see ENRICHER.md). The lm_head matmul is kept
    /// out of the graph on purpose (the frozen head needs no grads).
    pub loss: Tensor,
    /// True transition-paired masked CE value (detached; logging only).
    pub ce_value: f64,
    /// Scalar: mean pre-gate relative injection energy
    /// ||W shat||^2 / ||q||^2 over all steps (the penalty target).
    pub pen: Tensor,
    /// Mean applied (post-gate) relative injection energy, prompt positions.
    pub applied_prompt: f64,
    /// Mean applied relative injection energy, answer slots.
    pub applied_answer: f64,
    /// Detached transition logits [B, answer_len, vocab] (argmax / sampling
    /// only — not part of the backward graph).
    pub logits: Tensor,
}

/// Hard sample one token id from softmax(logits / temperature).
fn sample_from_logits(lg: &[f32], temperature: f64, rng: &mut rand::rngs::StdRng) -> u32 {
    let max = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let t = temperature.max(1e-3);
    let exps: Vec<f64> = lg.iter().map(|&l| ((l - max) as f64 / t).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut u = rng.random::<f64>() * sum;
    for (i, e) in exps.iter().enumerate() {
        u -= e;
        if u <= 0.0 {
            return i as u32;
        }
    }
    (exps.len() - 1) as u32
}

impl DecoderLayer {
    /// Cache-free single-position cascade step for the training forward:
    /// one query position attending over an explicit prefix KV (the previous
    /// cascade steps' keys/values, already RoPE'd). `h` is [B, hidden].
    /// Returns the new [B, hidden] residual plus this position's (k, v)
    /// ([B, num_kv_heads, 1, head_dim]) for the caller to append to the
    /// prefix. Causal by construction; callers must keep the sequence below
    /// the sliding window so no window mask is needed.
    ///
    /// With `kv_grad = false` the keys/values are detached *before* the
    /// attention cat: the KV-path gradient (both the own-position value
    /// write and the cross-position skip path) is dropped and the growing
    /// cat/expand copies never enter the autograd graph. The immediate
    /// gradient survives via the residual stream and the query path; the
    /// chain gradient survives via the state. This is the memory-safe
    /// default (the full KV path holds O(layers * tau * seq) activations
    /// and OOM-killed a 14 GB machine at tau=16; see ENRICHER.md).
    pub(super) fn cascade_train_step(
        &self,
        h: &Tensor,
        pos: usize,
        prefix: &[(Tensor, Tensor)],
        kv_grad: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, hidden) = h.dims2()?;
        let attn = &self.self_attn;
        let residual = h.clone();
        let xs = self.input_layernorm.forward(h)?;
        let x = xs.reshape((b_sz, 1, hidden))?;

        let q = attn
            .q_proj
            .forward(&x)?
            .reshape((b_sz, 1, attn.num_heads, attn.head_dim))?
            .transpose(1, 2)?;
        let k = attn
            .k_proj
            .forward(&x)?
            .reshape((b_sz, 1, attn.num_kv_heads, attn.head_dim))?
            .transpose(1, 2)?;
        let v = attn
            .v_proj
            .forward(&x)?
            .reshape((b_sz, 1, attn.num_kv_heads, attn.head_dim))?
            .transpose(1, 2)?;
        let q = attn.q_norm.forward(&q)?;
        let k = attn.k_norm.forward(&k)?;
        let (q, k) = attn.rotary_emb.apply_rotary_emb_qkv(&q, &k, pos)?;
        let (k, v) = if kv_grad {
            (k, v)
        } else {
            (k.detach(), v.detach())
        };

        let ks: Vec<&Tensor> = prefix
            .iter()
            .map(|(pk, _)| pk)
            .chain(std::iter::once(&k))
            .collect();
        let vs: Vec<&Tensor> = prefix
            .iter()
            .map(|(_, pv)| pv)
            .chain(std::iter::once(&v))
            .collect();
        let key_all = Tensor::cat(&ks, 2)?; // [B, nkv, t+1, hd]
        let value_all = Tensor::cat(&vs, 2)?;

        let key_exp = repeat_kv(key_all, attn.num_kv_groups)?;
        let value_exp = repeat_kv(value_all, attn.num_kv_groups)?;

        let scale = 1f64 / f64::sqrt(attn.query_pre_attn_scalar as f64);
        let attn_weights = (q.matmul(&key_exp.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attn.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights
            .matmul(&value_exp)?
            .transpose(1, 2)?
            .reshape((b_sz, ()))?
            .apply(&attn.o_proj)?;

        let xs = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = xs.clone();
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        let out = (residual + xs)?;
        Ok((out, k, v))
    }
}

impl Model {
    /// Sequential cascade training forward (TBPTT), matching inference
    /// exactly: lockstep batch processing of positions 0..T-1, the enricher
    /// active at every position, each position's input built from the
    /// previous step's state. Only the last `tau` positions are autograd
    /// connected (earlier enriched inputs are detached); the loss is the
    /// transition-paired masked CE over the answer slots, computed by the
    /// caller from `out.logits`.
    ///
    /// Fed tokens: positions < `answer_start` are the forced prompt tokens;
    /// answer slots feed the target token with probability `forced_prob`
    /// (CE warmup: 1.0), otherwise a token hard-sampled from this cascade's
    /// own logits at the previous step (temperature `gt`).
    #[allow(clippy::too_many_arguments)]
    pub fn cascade_train_forward(
        &self,
        enr: &Enricher,
        input_ids: &Tensor,
        targets: &Tensor,
        answer_start: usize,
        tau: usize,
        kv_grad: bool,
        forced_prob: f64,
        gt: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<CascadeTrainOut> {
        let (b, t_total) = input_ids.dims2()?;
        if answer_start == 0 {
            candle::bail!("cascade_train_forward needs a non-empty prompt");
        }
        let answer_len = targets.dim(1)?;
        if answer_start + answer_len > t_total {
            candle::bail!(
                "answer range exceeds the sequence ({answer_start}+{answer_len} > {t_total})"
            );
        }
        if t_total > self.sliding_window {
            candle::bail!(
                "cascade_train_forward assumes seq ({t_total}) < sliding window ({})",
                self.sliding_window
            );
        }
        if tau < answer_len + 1 {
            eprintln!(
                "  warn: tau={tau} < answer_len+1={}; transitions older than the window lose their state-chain gradient",
                answer_len + 1
            );
        }
        let ids = input_ids.to_vec2::<u32>()?;
        let targets_v = targets.to_vec2::<i64>()?;
        let vocab = self.lm_head.weight().dim(0)?;
        let n_valid = targets_v.iter().flatten().filter(|&&t| t >= 0).count() as f64;
        if n_valid == 0.0 {
            candle::bail!("no valid targets in batch");
        }
        let mut prev_logits: Option<Vec<Vec<f32>>> = None;
        let mut states: Option<Tensor> = None;
        let mut kvs: Vec<Vec<(Tensor, Tensor)>> = vec![Vec::new(); self.layers.len()];
        let mut trans: Vec<Tensor> = Vec::new();
        let mut loss_parts: Vec<Tensor> = Vec::new();
        let mut pens: Vec<Tensor> = Vec::new();
        let mut ce_sum = 0f64;
        let mut appl_prompt_sum = 0f64;
        let mut appl_answer_sum = 0f64;
        let mut appl_prompt_n = 0usize;
        let mut appl_answer_n = 0usize;
        let no_grad_until = t_total.saturating_sub(tau);
        let cascade_t0 = std::time::Instant::now();
        for t in 0..t_total {
            if t % 16 == 0 {
                eprintln!(
                    "  [cascade t={t}/{t_total}] elapsed {:?}",
                    cascade_t0.elapsed()
                );
            }
            // Fed token ids for this position.
            let fed: Vec<u32> = if t < answer_start || forced_prob >= 1.0 {
                ids.iter().map(|r| r[t]).collect()
            } else {
                let pl = prev_logits
                    .as_ref()
                    .expect("sampling without previous logits");
                (0..b)
                    .map(|r| {
                        if rng.random::<f64>() < forced_prob {
                            ids[r][t]
                        } else {
                            sample_from_logits(&pl[r], gt, rng)
                        }
                    })
                    .collect()
            };
            let fed_t = Tensor::from_vec(fed, (b, 1), &self.device)?;
            // Detached on purpose: the frozen embedding table needs no
            // grads (the enricher gradient flows through the injection
            // branch of x = q + g*W*shat); keeping the table in the graph
            // would allocate a [vocab, hidden] grad at backward.
            let e = self
                .embed_tokens
                .forward(&fed_t)?
                .to_dtype(DType::F32)?
                .squeeze(1)?
                .detach(); // [B, H] unscaled rows
            let (x, preg, appl) = enr.enrich(&e, states.as_ref())?;
            let x = if t < no_grad_until { x.detach() } else { x };
            let appl_m = appl.sum_all()?.to_scalar::<f32>()? as f64 / b as f64;
            if t < answer_start {
                appl_prompt_sum += appl_m;
                appl_prompt_n += 1;
            } else {
                appl_answer_sum += appl_m;
                appl_answer_n += 1;
            }
            pens.push(preg.mean_all()?);
            let mut h = x.to_dtype(self.dtype)?;
            for (li, layer) in self.layers.iter().enumerate() {
                let (h_new, k, v) = layer.cascade_train_step(&h, t, &kvs[li], kv_grad)?;
                kvs[li].push((k, v));
                h = h_new;
            }
            // Transition slot k: logits at this step predict targets[:, k].
            if t + 1 >= answer_start && t + 1 < answer_start + answer_len {
                let k = t + 1 - answer_start;
                let normed = h.unsqueeze(1)?.apply(&self.norm)?; // [B, 1, H] graph op
                // ---- Detached head branch (never traversed by backward;
                // the frozen head needs no grads and its [V, H] grad buffer
                // is the single largest allocation in the graph) ----
                if self.final_logit_softcapping.is_some() {
                    candle::bail!("cascade_train_forward does not support final_logit_softcapping");
                }
                let logits_det = normed.detach().apply(&self.lm_head)?; // [B, 1, V]
                let lg = logits_det.squeeze(1)?.to_dtype(DType::F32)?; // [B, V]
                let p = candle_nn::ops::softmax_last_dim(&lg)?;
                let tgt: Vec<i64> = (0..b).map(|r| targets_v[r][k]).collect();
                let idx = Tensor::from_vec(
                    tgt.iter()
                        .map(|&t| if t >= 0 { t as u32 } else { 0u32 })
                        .collect(),
                    (b,),
                    &self.device,
                )?;
                let picked = p.gather(&idx.unsqueeze(1)?, 1)?.reshape((b,))?;
                let mask_t = Tensor::from_vec(
                    tgt.iter()
                        .map(|&t| if t >= 0 { 1.0f32 } else { 0.0 })
                        .collect(),
                    (b,),
                    &self.device,
                )?;
                ce_sum += -((picked.log()? * &mask_t)?.sum_all()?.to_scalar::<f32>()? as f64);
                // d(CE_slot)/d(normed) = (softmax - onehot) / n_valid @ W_head,
                // built detached and injected via sum(normed * g).
                let mut oh = vec![0.0f32; b * vocab];
                for (r, &tv) in tgt.iter().enumerate() {
                    if tv >= 0 {
                        oh[r * vocab + tv as usize] = 1.0;
                    }
                }
                let onehot = Tensor::from_vec(oh, (b, vocab), &self.device)?;
                let dlog = ((&p - &onehot)?.affine(1.0 / n_valid, 0.0))?;
                let g_dn = dlog.matmul(self.lm_head.weight())?; // [B, H] detached
                let g_dn = g_dn
                    .reshape((b, 1, self.hidden_size))?
                    .to_dtype(normed.dtype())?
                    .broadcast_mul(&mask_t.reshape((b, 1, 1))?)?; // zero on pads
                if t >= no_grad_until {
                    loss_parts.push((normed.broadcast_mul(&g_dn))?.sum_all()?);
                }
                if forced_prob < 1.0 && t + 1 < t_total {
                    prev_logits = Some(lg.to_vec2::<f32>()?);
                }
                trans.push(logits_det);
            }
            states = Some(h);
        }
        let logits = Tensor::cat(&trans, 1)?; // [B, A, V], detached
        let pen = Tensor::stack(&pens, 0)?.mean_all()?;
        let loss = if loss_parts.is_empty() {
            Tensor::new(0.0f32, &self.device)?
        } else {
            Tensor::stack(&loss_parts, 0)?.sum_all()?
        };
        Ok(CascadeTrainOut {
            loss,
            ce_value: ce_sum / n_valid,
            pen,
            applied_prompt: appl_prompt_sum / appl_prompt_n.max(1) as f64,
            applied_answer: appl_answer_sum / appl_answer_n.max(1) as f64,
            logits,
        })
    }
}
