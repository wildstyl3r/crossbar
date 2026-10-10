use std::io::{Read, Seek};
use std::sync::Arc;

use candle::quantized::{QMatMul as InnerQMatMul, QTensor, gguf_file};
use candle::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, kv_cache::KvCache, kv_cache::RotatingKvCache};

use crate::model::{
    Config, FeedbackConfig, FeedbackStats, FeedbackTempMode, KvCacheKind, PrefillStats, ProbeOut,
    RmsNorm, RotaryEmbedding, prepare_decoder_attention_mask, repeat_kv, solve_entropy_temperature,
};

struct QMatMul {
    inner: InnerQMatMul,
}

impl QMatMul {
    fn from_qtensor(qt: QTensor) -> Result<Self> {
        Ok(Self {
            inner: InnerQMatMul::from_qtensor(qt)?,
        })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

fn md_u32(ct: &gguf_file::Content, key: &str) -> Option<u32> {
    ct.metadata
        .get(&format!("gemma3.{key}"))
        .and_then(|v| v.to_u32().ok())
}

fn md_f32(ct: &gguf_file::Content, key: &str) -> Option<f32> {
    ct.metadata
        .get(&format!("gemma3.{key}"))
        .and_then(|v| v.to_f32().ok())
}

fn req_u32(ct: &gguf_file::Content, key: &str) -> Result<u32> {
    md_u32(ct, key)
        .ok_or_else(|| candle::Error::Msg(format!("missing gemma3.{key} in GGUF metadata")))
}

fn norm_from_gguf(t: QTensor, eps: f64, device: &Device) -> Result<RmsNorm> {
    let weight = (t.dequantize(device)?.to_dtype(DType::F32)? - 1.0f64)?;
    Ok(RmsNorm::from_weight(weight, eps))
}

fn config_from_gguf(
    ct: &gguf_file::Content,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<Config> {
    let key_length = req_u32(ct, "attention.key_length")? as usize;
    Ok(Config {
        attention_bias: false,
        head_dim: key_length,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size,
        intermediate_size: req_u32(ct, "feed_forward_length")? as usize,
        num_attention_heads: req_u32(ct, "attention.head_count")? as usize,
        num_hidden_layers: req_u32(ct, "block_count")? as usize,
        num_key_value_heads: req_u32(ct, "attention.head_count_kv")? as usize,
        rms_norm_eps: md_f32(ct, "attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        rope_theta: md_f32(ct, "rope.freq_base").unwrap_or(1_000_000.0) as f64,
        rope_local_base_freq: md_f32(ct, "rope.local_freq_base").unwrap_or(10_000.0) as f64,
        vocab_size,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: key_length,
        sliding_window: req_u32(ct, "attention.sliding_window")? as usize,
        sliding_window_pattern: md_u32(ct, "attention.sliding_window_type").unwrap_or(6) as usize,
        max_position_embeddings: md_u32(ct, "context_length").unwrap_or(32768) as usize,
        layer_types: None,
    })
}

struct QuantizedAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    query_pre_attn_scalar: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCacheKind,
}

impl QuantizedAttention {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &mut self.kv_cache {
            KvCacheKind::Normal(cache) => cache.append(&key_states, &value_states)?,
            KvCacheKind::Rotating(cache) => cache.append(&key_states, &value_states)?,
        };

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.query_pre_attn_scalar as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match self.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCacheKind::Normal(c) => c.reset(),
            KvCacheKind::Rotating(c) => c.reset(),
        }
    }
}

struct QuantizedMlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
}

impl Module for QuantizedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct QuantizedDecoderLayer {
    self_attn: QuantizedAttention,
    mlp: QuantizedMlp,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}

impl QuantizedDecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

pub struct QuantizedModel {
    embed_tokens: Embedding,
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    max_positions: usize,
    sliding_window: usize,
    feedback: FeedbackConfig,
    /// Current temperature for the Heuristic mode; reset by clear_kv_cache.
    heuristic_t: f64,
    /// Previous fed-back input and its norm, for the consecutive cosine.
    last_input: Option<(Tensor, f64)>,
    /// Optional trained attention-pooling enricher; when present it replaces
    /// the softmax-blend formula for the decode input.
    pub enricher: Option<crate::enrich::Enricher>,
    /// F32 copy of the embedding table, built lazily for the softmax
    /// feedback (the stored table is BF16).
    embeddings_f32: Option<Tensor>,
}

impl QuantizedModel {
    pub fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        device: &Device,
        feedback: FeedbackConfig,
    ) -> Result<Self> {
        let ct = gguf_file::Content::read(reader)?;
        let token_embd = ct.tensor(reader, "token_embd.weight", device)?;
        let dims = token_embd.shape().dims();
        if dims.len() != 2 {
            candle::bail!("unexpected token_embd.weight shape {dims:?}");
        }
        let (vocab_size, hidden_size) = (dims[0], dims[1]);
        let cfg = config_from_gguf(&ct, vocab_size, hidden_size)?;

        let embed_tokens = Embedding::new(
            token_embd.dequantize(device)?.to_dtype(DType::BF16)?,
            cfg.hidden_size,
        );
        let norm = norm_from_gguf(
            ct.tensor(reader, "output_norm.weight", device)?,
            cfg.rms_norm_eps,
            device,
        )?;
        let lm_head = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => QMatMul::from_qtensor(t)?,
            Err(_) => QMatMul::from_qtensor(token_embd)?,
        };

        let rotary_full = Arc::new(RotaryEmbedding::new(DType::F32, &cfg, device, None)?);
        let rotary_sliding = Arc::new(RotaryEmbedding::new(
            DType::F32,
            &cfg,
            device,
            Some(cfg.sliding_window),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let p = format!("blk.{layer_idx}");
            let q_proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?;
            let k_proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?)?;
            let v_proj =
                QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?)?;
            let o_proj = QMatMul::from_qtensor(ct.tensor(
                reader,
                &format!("{p}.attn_output.weight"),
                device,
            )?)?;
            let q_norm = norm_from_gguf(
                ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                cfg.rms_norm_eps,
                device,
            )?;
            let k_norm = norm_from_gguf(
                ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                cfg.rms_norm_eps,
                device,
            )?;
            let mlp = QuantizedMlp {
                gate_proj: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_gate.weight"),
                    device,
                )?)?,
                up_proj: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_up.weight"),
                    device,
                )?)?,
                down_proj: QMatMul::from_qtensor(ct.tensor(
                    reader,
                    &format!("{p}.ffn_down.weight"),
                    device,
                )?)?,
                act_fn: cfg.hidden_activation,
            };
            let input_layernorm = norm_from_gguf(
                ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                cfg.rms_norm_eps,
                device,
            )?;
            let post_attention_layernorm = norm_from_gguf(
                ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                cfg.rms_norm_eps,
                device,
            )?;
            let pre_feedforward_layernorm = norm_from_gguf(
                ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                cfg.rms_norm_eps,
                device,
            )?;
            let post_feedforward_layernorm = norm_from_gguf(
                ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                cfg.rms_norm_eps,
                device,
            )?;

            let sliding = cfg.is_sliding_layer(layer_idx);
            let rotary_emb = if sliding {
                rotary_sliding.clone()
            } else {
                rotary_full.clone()
            };
            let kv_cache = if sliding {
                KvCacheKind::Rotating(RotatingKvCache::new(2, cfg.sliding_window))
            } else {
                KvCacheKind::Normal(KvCache::new(2, cfg.max_position_embeddings))
            };
            layers.push(QuantizedDecoderLayer {
                self_attn: QuantizedAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    num_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads,
                    num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                    head_dim: cfg.head_dim,
                    query_pre_attn_scalar: cfg.query_pre_attn_scalar,
                    attn_logit_softcapping: cfg.attn_logit_softcapping,
                    rotary_emb,
                    kv_cache,
                },
                mlp,
                input_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                post_attention_layernorm,
                sliding_window: sliding.then_some(cfg.sliding_window),
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: device.clone(),
            dtype: DType::F32,
            hidden_size: cfg.hidden_size,
            max_positions: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
            feedback,
            heuristic_t: feedback.temperature,
            last_input: None,
            enricher: None,
            embeddings_f32: None,
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
        let xs = (xs.to_dtype(DType::F32)? * (self.hidden_size as f64).sqrt())?;

        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, seqlen_offset)?;

        let xs = self.run_layers(xs, &attention_mask, &sliding_attention_mask, seqlen_offset)?;
        let last = xs.narrow(1, seq_len - 1, 1)?;
        let out = self.finish(last.clone())?;
        if self.enricher.is_some() {
            let state = last.squeeze(1)?.squeeze(0)?.contiguous()?;
            if let Some(enr) = &mut self.enricher {
                enr.push_state(state);
            }
        }
        Ok(out)
    }

    /// Feed the previous step's blended input (expected token embedding +
    /// sampled token embedding) back into the first block, skipping the pure
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
            FeedbackTempMode::Heuristic => Ok(self.heuristic_t),
        }
    }

    fn ensure_embeddings_f32(&mut self) -> Result<()> {
        if self.embeddings_f32.is_none() {
            self.embeddings_f32 = Some(self.embed_tokens.embeddings().to_dtype(DType::F32)?);
        }
        Ok(())
    }

    /// Expected embedding from raw logits at the given temperature, as
    /// softmax(logits / T) @ E * sqrt(hidden_size). Always computed in F32
    /// against a dequantized copy of the embedding table.
    fn expected_from_logits(&self, logits_f32: &Tensor, temperature: f64) -> Result<Tensor> {
        let probs = candle_nn::ops::softmax_last_dim(&(logits_f32 / temperature)?)?;
        let (b_sz, seq_len, vocab) = probs.dims3()?;
        let embeddings = self.embeddings_f32.as_ref().unwrap();
        let xs = probs.reshape((b_sz * seq_len, vocab))?.matmul(embeddings)?;
        xs.reshape((b_sz, seq_len, ()))?
            .affine((self.hidden_size as f64).sqrt(), 0.0)
    }

    fn feedback_input(
        &mut self,
        logits: &Tensor,
        sampled_token: u32,
    ) -> Result<(Tensor, FeedbackStats)> {
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let temperature = self.resolve_temperature(&logits_f32, 1)?;
        self.ensure_embeddings_f32()?;
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
        if matches!(self.feedback.mode, FeedbackTempMode::Heuristic) {
            let smear = entropy > self.feedback.target_entropy + self.feedback.band
                || cosine_prev > self.feedback.max_cosine;
            let overconfident = entropy < self.feedback.target_entropy - self.feedback.band;
            let mut next = self.heuristic_t;
            if smear {
                next = (next * self.feedback.decay).max(1e-3);
            } else if overconfident {
                next = (next / self.feedback.decay).min(20.0);
            }
            self.heuristic_t = next;
        }
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
    /// expected embeddings from full-position logits, then one injection
    /// pass over the blended embeddings builds the KV cache the decode loop
    /// continues from. Returns last-position logits and summary stats.
    pub fn forward_prompt_blend(&mut self, input_ids: &Tensor) -> Result<(Tensor, PrefillStats)> {
        let (b_size, seq_len) = input_ids.dims2()?;
        const CHUNK: usize = 64;
        let alpha = self.feedback.prefill_blend.clamp(0.0, 1.0);
        if self.feedback.prefill_iters > 1 {
            candle::bail!("iterated prefill feedback (cascade) requires the safetensors backend");
        }
        let mut expected_chunks: Vec<Tensor> = Vec::new();
        let mut t_sum = 0f64;
        let mut h_sum = 0f64;
        let mut n_positions = 0usize;
        self.ensure_embeddings_f32()?;
        for start in (0..seq_len).step_by(CHUNK) {
            let len = CHUNK.min(seq_len - start);
            let ids = input_ids.narrow(1, start, len)?;
            let xs = self
                .embed_tokens
                .forward(&ids)?
                .to_dtype(DType::F32)?
                .affine((self.hidden_size as f64).sqrt(), 0.0)?;
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
        self.reset_layer_caches();

        let mut chunks = Vec::with_capacity(expected_chunks.len());
        for (ci, expected) in expected_chunks.into_iter().enumerate() {
            let start = ci * CHUNK;
            let len = CHUNK.min(seq_len - start);
            let ids = input_ids.narrow(1, start, len)?;
            let base = self
                .embed_tokens
                .forward(&ids)?
                .to_dtype(DType::F32)?
                .affine((self.hidden_size as f64).sqrt(), 0.0)?;
            let blended = if alpha >= 1.0 {
                expected
            } else {
                (&base * (1.0 - alpha))?.broadcast_add(&(&expected * alpha)?)?
            };
            chunks.push(blended.to_dtype(self.dtype)?);
        }
        let xs = Tensor::cat(&chunks, 1)?;
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(b_size, seq_len, 0)?;
        let xs = self.run_layers(xs, &attention_mask, &sliding_attention_mask, 0)?;
        let stats = PrefillStats {
            temperature: if n_positions > 0 {
                t_sum / (seq_len as f64 / CHUNK as f64).ceil()
            } else {
                f64::NAN
            },
            entropy: if n_positions > 0 {
                h_sum / n_positions as f64
            } else {
                f64::NAN
            },
            iters: 1,
            entropies: vec![if n_positions > 0 {
                h_sum / n_positions as f64
            } else {
                f64::NAN
            }],
        };
        let last = xs.narrow(1, seq_len - 1, 1)?;
        let logits = self.finish(last.clone())?;
        if self.enricher.is_some() {
            let state = last.squeeze(1)?.squeeze(0)?.contiguous()?;
            if let Some(enr) = &mut self.enricher {
                enr.push_state(state);
            }
        }
        Ok((logits, stats))
    }

    fn reset_layer_caches(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    /// Attach a trained enricher (inference); replaces the softmax-blend
    /// decode feedback with the learned state-readout enrichment.
    pub fn set_enricher(&mut self, enr: crate::enrich::Enricher) {
        self.enricher = Some(enr);
    }

    /// Whether the enricher is attached (decides the cascade inference path).
    pub fn has_enricher(&self) -> bool {
        self.enricher.is_some()
    }

    /// One step of the sequential cascade with the enricher attached:
    /// enrich the forced/sampled token's embedding from the previous state,
    /// forward a single position at `seqlen_offset`, push the new state,
    /// return the logits.
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
        self.heuristic_t = self.feedback.temperature;
        self.last_input = None;
        if let Some(enr) = &mut self.enricher {
            enr.clear_state();
        }
    }

    pub fn probe(&mut self, token: u32) -> Result<ProbeOut> {
        let row = self
            .embed_tokens
            .embeddings()
            .narrow(0, token as usize, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?;
        let row_v = row.to_vec1::<f32>()?;
        let norm = (row_v.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt();
        let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
        let logits = self.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&i, &j| logits[j].total_cmp(&logits[i]));
        let top5 = order
            .iter()
            .take(5)
            .map(|&i| (i as u32, logits[i]))
            .collect();
        Ok(((row_v[..5].to_vec(), norm), top5))
    }
}

pub fn probe_weights(path: &str, device: &Device) -> Result<()> {
    let mut reader = std::fs::File::open(path)?;
    let ct = gguf_file::Content::read(&mut reader)?;
    for name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_gate.weight",
        "token_embd.weight",
    ] {
        let t = ct.tensor(&mut reader, name, device)?;
        let flat = t.dequantize(device)?.to_dtype(DType::F32)?.flatten_all()?;
        let n = flat.elem_count().min(8);
        let vals = flat.narrow(0, 0, n)?.to_vec1::<f32>()?;
        println!("gguf {name}[:{n}] = {vals:?}");
    }
    Ok(())
}
