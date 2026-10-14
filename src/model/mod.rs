//! Gemma-3 safetensors model (F32 on CPU, BF16 on CUDA): architecture
//! wiring and inference paths (plain forward, softmax feedback, prefill
//! cascade, enricher cascade).

pub(crate) mod layers;

use crate::config::Config;
use crate::experiments::feedback::solve_entropy_temperature;
use crate::experiments::feedback::{
    FeedbackConfig, FeedbackStats, FeedbackTempMode, PrefillStats,
};
use layers::{RmsNorm, RotaryEmbedding, prepare_decoder_attention_mask};

use std::sync::Arc;

use candle::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::experiments::enrich;
use layers::DecoderLayer;

pub struct Model {
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) norm: RmsNorm,
    pub(crate) lm_head: Linear,
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    pub(crate) hidden_size: usize,
    max_positions: usize,
    pub(crate) sliding_window: usize,
    feedback: FeedbackConfig,
    /// Previous fed-back input and its norm, for the consecutive cosine.
    last_input: Option<(Tensor, f64)>,
    /// Optional trained attention-pooling enricher; when present it replaces
    /// the softmax-blend formula for the decode input.
    pub enricher: Option<enrich::Enricher>,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder, feedback: FeedbackConfig) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_full = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device(), None)?);
        let rotary_sliding = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb.device(),
            Some(cfg.sliding_window),
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let sliding = cfg.is_sliding_layer(layer_idx);
            let rotary_emb = if sliding {
                rotary_sliding.clone()
            } else {
                rotary_full.clone()
            };
            let layer = DecoderLayer::new(
                rotary_emb,
                cfg,
                vb_l.pp(layer_idx),
                sliding.then_some(cfg.sliding_window),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            max_positions: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            feedback,
            last_input: None,
            enricher: None,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }

        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;

        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;

        Ok((Some(mask), Some(sliding_mask)))
    }

    fn run_layers(
        &mut self,
        mut xs: Tensor,
        attention_mask: &Option<Tensor>,
        sliding_attention_mask: &Option<Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        for layer in self.layers.iter_mut() {
            let mask = if layer.sliding_window.is_some() {
                sliding_attention_mask
            } else {
                attention_mask
            };
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset)?
        }
        Ok(xs)
    }

    fn finish(&self, last: Tensor) -> Result<Tensor> {
        let logits = last.apply(&self.norm)?.apply(&self.lm_head)?;
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };
        Ok(logits)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = (xs * (self.hidden_size as f64).sqrt())?;

        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset)?;

        let xs = self.run_layers(xs, &attention_mask, &sliding_attention_mask, seqlen_offset)?;
        let last = xs.narrow(1, seq_len - 1, 1)?;
        let out = self.finish(last.clone())?;
        if let Some(enr) = &mut self.enricher {
            let state = last.squeeze(1)?.squeeze(0)?.contiguous()?;
            enr.push_state(state);
        }
        Ok(out)
    }

    /// Whether the enricher is attached (decides the cascade inference path).
    pub fn has_enricher(&self) -> bool {
        self.enricher.is_some()
    }

    /// One step of the sequential cascade with the enricher attached:
    /// enrich the forced/sampled token's embedding from the previous state,
    /// forward a single position at `seqlen_offset`, push the new state,
    /// return the logits. Used for both the sequential forced-token prefill
    /// (one call per prompt token) and every decode step — the parallel
    /// prefill is never used while the enricher is attached.
    pub fn cascade_step(&mut self, token: u32, seqlen_offset: usize) -> Result<Tensor> {
        if self.enricher.is_none() {
            candle::bail!("cascade_step requires an attached enricher");
        }
        let input = {
            let enr = self.enricher.as_ref().unwrap();
            let token_t = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let e = self
                .embed_tokens
                .forward(&token_t)?
                .to_dtype(DType::F32)?
                .squeeze(0)?
                .squeeze(0)?;
            let (input, _) = enr.enrich_1d(&e, enr.prev_state())?;
            input.to_dtype(self.dtype)?.unsqueeze(0)?.unsqueeze(0)?
        };
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(1, 1, seqlen_offset)?;
        let xs = self.run_layers(
            input,
            &attention_mask,
            &sliding_attention_mask,
            seqlen_offset,
        )?;
        let last = xs.narrow(1, 0, 1)?;
        let out = self.finish(last.clone())?;
        if let Some(enr) = &mut self.enricher {
            enr.push_state(last.squeeze(1)?.squeeze(0)?.contiguous()?);
        }
        Ok(out)
    }

    /// Feed the previous step's blended input ((1 - a) sampled-token +
    /// a expected embedding) back into the first block, skipping the pure
    /// embedding lookup of the sampled token. Also resolves the feedback
    /// temperature for the configured mode and returns the per-step
    /// feedback statistics.
    pub fn forward_feedback(
        &mut self,
        logits: &Tensor,
        sampled_token: u32,
        seqlen_offset: usize,
    ) -> Result<(Tensor, FeedbackStats)> {
        if self.enricher.is_some() {
            let out = self.cascade_step(sampled_token, seqlen_offset)?;
            let stats = FeedbackStats {
                temperature: f64::NAN,
                entropy: f64::NAN,
                support: f64::NAN,
                input_norm: f64::NAN,
                cosine_prev: f64::NAN,
            };
            return Ok((out, stats));
        }
        let (xs, stats) = self.feedback_input(logits, sampled_token)?;
        let (b_size, seq_len, _) = xs.dims3()?;

        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset)?;

        let xs = self.run_layers(xs, &attention_mask, &sliding_attention_mask, seqlen_offset)?;
        Ok((self.finish(xs)?, stats))
    }

    fn resolve_temperature(&self, logits_f32: &Tensor, n_positions: usize) -> Result<f64> {
        match self.feedback.mode {
            FeedbackTempMode::Fixed => Ok(self.feedback.temperature),
            FeedbackTempMode::EntropyServo => solve_entropy_temperature(
                logits_f32,
                self.feedback.target_entropy * n_positions as f64,
            ),
            FeedbackTempMode::OneSidedServo => solve_entropy_temperature(
                logits_f32,
                self.feedback.target_entropy * n_positions as f64,
            )
            .map(|t| t.min(1.0)),
        }
    }

    /// Expected embedding from raw logits at the given temperature, as
    /// softmax(logits / T) @ E * sqrt(hidden_size).
    fn expected_from_logits(&self, logits_f32: &Tensor, temperature: f64) -> Result<Tensor> {
        let probs = candle_nn::ops::softmax_last_dim(&(logits_f32 / temperature)?)?;
        let (b_sz, seq_len, vocab) = probs.dims3()?;
        let embeddings = self.embed_tokens.embeddings();
        let xs = probs
            .reshape((b_sz * seq_len, vocab))?
            .to_dtype(embeddings.dtype())?
            .matmul(embeddings)?;
        xs.reshape((b_sz, seq_len, ()))?
            .affine((self.hidden_size as f64).sqrt(), 0.0)?
            .to_dtype(DType::F32)
    }

    fn feedback_input(
        &mut self,
        logits: &Tensor,
        sampled_token: u32,
    ) -> Result<(Tensor, FeedbackStats)> {
        if self.enricher.is_some() {
            unreachable!("feedback_input is not used when the enricher is active");
        }
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let temperature = self.resolve_temperature(&logits_f32, 1)?;
        let expected = self.expected_from_logits(&logits_f32, temperature)?;
        let probs = candle_nn::ops::softmax_last_dim(&(logits_f32 / temperature)?)?;
        let logp = (&probs + 1e-12)?.log()?;
        let entropy = -((&probs * &logp)?.sum_all()?.to_scalar::<f32>()? as f64);

        let alpha = self.feedback.blend;
        let xs = if alpha >= 1.0 {
            expected
        } else {
            let token = Tensor::new(&[sampled_token], &self.device)?.unsqueeze(0)?;
            let row = self
                .embed_tokens
                .forward(&token)?
                .to_dtype(DType::F32)?
                .affine((self.hidden_size as f64).sqrt(), 0.0)?;
            (&row * (1.0 - alpha))?.broadcast_add(&(&expected * alpha)?)?
        };
        let xs = xs.to_dtype(self.dtype)?;
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let norm = xs_f32
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
        let cosine_prev = match &self.last_input {
            Some((prev, prev_norm)) => {
                let dot = (&xs_f32 * prev)?.sum_all()?.to_scalar::<f32>()? as f64;
                if *prev_norm > 0.0 && norm > 0.0 {
                    dot / (prev_norm * norm)
                } else {
                    f64::NAN
                }
            }
            None => f64::NAN,
        };
        self.last_input = Some((xs_f32, norm));
        let stats = FeedbackStats {
            temperature,
            entropy,
            support: entropy.exp(),
            input_norm: norm,
            cosine_prev,
        };
        Ok((xs, stats))
    }

    /// Prefill with prompt feedback: an observation pass builds per-position
    /// expected embeddings from full-position logits, then `prefill_iters`
    /// refinement iterations re-forward the blended inputs
    /// `(1 - a) * E_prompt + a * expected(logits_{k-1})` over all positions
    /// with fresh caches; the last iteration's cache is the one the decode
    /// loop continues from. Returns last-position logits and summary stats.
    pub fn forward_prompt_blend(&mut self, input_ids: &Tensor) -> Result<(Tensor, PrefillStats)> {
        let (b_size, seq_len) = input_ids.dims2()?;
        const CHUNK: usize = 64;
        let alpha = self.feedback.prefill_blend.clamp(0.0, 1.0);
        let iters = self.feedback.prefill_iters.max(1);

        // Scaled prompt embeddings E * sqrt(h) in F32: the base input every
        // iteration blends against. Gathered once for the whole prompt
        // (per-chunk gathers would return the same rows).
        let base = self
            .embed_tokens
            .forward(input_ids)?
            .affine((self.hidden_size as f64).sqrt(), 0.0)?
            .to_dtype(DType::F32)?;

        // ---- Iteration 0: chunked observation pass. Caches accumulate
        // across chunks and are discarded afterwards; per-chunk full-position
        // logits yield the iteration-0 expected embeddings. ----
        let mut expected_chunks: Vec<Tensor> = Vec::new();
        let mut t_sum = 0f64;
        let mut h_sum = 0f64;
        let mut n_positions = 0usize;
        for start in (0..seq_len).step_by(CHUNK) {
            let len = CHUNK.min(seq_len - start);
            let xs = base.narrow(1, start, len)?;
            let (attention_mask, sliding_attention_mask) =
                self.create_attention_masks(b_size, len, start)?;
            let xs = self.run_layers(xs, &attention_mask, &sliding_attention_mask, start)?;
            let logits_f32 = self.finish(xs)?.to_dtype(DType::F32)?;
            let temperature = self.resolve_temperature(&logits_f32, len)?;
            let probs = candle_nn::ops::softmax_last_dim(&(&logits_f32 / temperature)?)?;
            let logp = (&probs + 1e-12)?.log()?;
            let chunk_h =
                -((&probs * &logp)?.sum_all()?.to_scalar::<f32>()? as f64) / (b_size * len) as f64;
            t_sum += temperature;
            h_sum += chunk_h * len as f64;
            n_positions += len;
            expected_chunks.push(self.expected_from_logits(&logits_f32, temperature)?);
        }
        let stats_temperature = t_sum / (seq_len as f64 / CHUNK as f64).ceil();
        let mut entropies = vec![h_sum / n_positions as f64];

        // ---- Refinement iterations 1..=K. Each re-forwards
        // (1 - a_p) * base + a_p * expected_{k-1} over all positions with
        // fresh caches. Iterations before the last also produce per-position
        // logits -> expected embeddings for the next iteration. ----
        let mut expected = Tensor::cat(&expected_chunks, 1)?;
        for k in 1..=iters {
            let inputs = if alpha >= 1.0 {
                expected.clone()
            } else {
                (&base * (1.0 - alpha))?.broadcast_add(&(&expected * alpha)?)?
            };
            self.reset_layer_caches();
            let (attention_mask, sliding_attention_mask) =
                self.create_attention_masks(b_size, seq_len, 0)?;
            let xs = self.run_layers(
                inputs.to_dtype(self.dtype)?,
                &attention_mask,
                &sliding_attention_mask,
                0,
            )?;
            if k < iters {
                // Per-position logits -> expected embeddings for the next
                // iteration. (Full-sequence temperature resolution; servo
                // targets scale with the total length.)
                let logits_f32 = self.finish(xs)?.to_dtype(DType::F32)?;
                let temperature = self.resolve_temperature(&logits_f32, seq_len)?;
                let probs = candle_nn::ops::softmax_last_dim(&(&logits_f32 / temperature)?)?;
                let logp = (&probs + 1e-12)?.log()?;
                entropies.push(
                    -((&probs * &logp)?.sum_all()?.to_scalar::<f32>()? as f64)
                        / (b_size * seq_len) as f64,
                );
                expected = self.expected_from_logits(&logits_f32, temperature)?;
            } else {
                // The decode loop continues from this iteration's cache.
                let last = xs.narrow(1, seq_len - 1, 1)?;
                let logits = self.finish(last.clone())?;
                if let Some(enr) = &mut self.enricher {
                    let state = last.squeeze(1)?.squeeze(0)?.contiguous()?;
                    enr.push_state(state);
                }
                let stats = PrefillStats {
                    temperature: stats_temperature,
                    entropy: entropies[0],
                    iters,
                    entropies,
                };
                return Ok((logits, stats));
            }
        }
        unreachable!("prefill cascade loop must return at the final iteration")
    }

    fn reset_layer_caches(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    /// Attach a trained enricher (inference); replaces the softmax-blend
    /// decode feedback with the learned attention-pooling enrichment.
    pub fn set_enricher(&mut self, enr: enrich::Enricher) {
        self.enricher = Some(enr);
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    pub fn clear_kv_cache(&mut self) {
        self.reset_layer_caches();
        self.last_input = None;
        if let Some(enr) = &mut self.enricher {
            enr.clear_state();
        }
    }
}
