use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    linear_b as linear, Activation, Embedding, Linear, VarBuilder,
    kv_cache::{KvCache, RotatingKvCache},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub vocab_size: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub sliding_window: usize,
    #[serde(rename = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
}

impl Config {
    pub fn is_sliding_layer(&self, idx: usize) -> bool {
        match &self.layer_types {
            Some(types) if idx < types.len() => types[idx] == LayerType::SlidingAttention,
            _ => !(idx + 1).is_multiple_of(self.sliding_window_pattern),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub(crate) fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub(crate) fn new(
        dtype: DType,
        cfg: &Config,
        dev: &Device,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let rope_freq = if sliding_window.is_some() {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub(crate) fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub(crate) enum KvCacheKind {
    Normal(KvCache),
    Rotating(RotatingKvCache),
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        sliding_window: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let kv_cache = if let Some(sliding_window) = sliding_window {
            KvCacheKind::Rotating(RotatingKvCache::new(2, sliding_window))
        } else {
            KvCacheKind::Normal(KvCache::new(2, cfg.max_position_embeddings))
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            kv_cache,
        })
    }

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

    /// Cache-free attention for the training forward: full-sequence causal
    /// attention with an explicit (pad-aware) additive mask, no KV cache.
    fn forward_train(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
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
                .apply_rotary_emb_qkv(&query_states, &key_states, 0)?;

        let key_states = repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?;

        let scale = 1f64 / f64::sqrt(self.query_pre_attn_scalar as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match self.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        let attn_weights = attn_weights.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

pub(crate) fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
    }
}

/// Causal attention mask for the cache-free training forward, with per-row
/// padding: column j is masked unless j <= i, j < lens[b] and (for the
/// sliding variant) j + window >= i.
pub(crate) fn prepare_train_mask(
    batch: usize,
    seq_len: usize,
    lens: &[usize],
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut v = Vec::with_capacity(batch * seq_len * seq_len);
    for b in 0..batch {
        let len = lens.get(b).copied().unwrap_or(seq_len).min(seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let keep = j <= i
                    && j < len
                    && sliding_window.map_or(true, |w| j + w >= i);
                v.push(if keep { 0f32 } else { f32::NEG_INFINITY });
            }
        }
    }
    Tensor::from_vec(v, (batch, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sliding_window: Option<usize>,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, sliding_window, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            sliding_window,
        })
    }

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

    /// Cache-free decoder layer for the training forward.
    fn forward_train(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_train(&xs, mask)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

pub(crate) type ProbeOut = ((Vec<f32>, f64), Vec<(u32, f32)>);

pub(crate) fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FeedbackTempMode {
    /// Constant temperature from --softmax-feedback-temperature.
    Fixed,
    /// Per-step servo: solve T so that H(softmax(logits / T)) equals the
    /// target entropy.
    EntropyServo,
    /// Like EntropyServo, but the solution is capped at T <= 1: the servo
    /// only tames entropy explosions (T < 1) and otherwise falls back to the
    /// model's own distribution (T = 1) instead of blurring confident steps.
    OneSidedServo,
    /// Persistent temperature, multiplicatively decayed/grown when the
    /// measured feedback statistics leave the target entropy band or the
    /// consecutive inputs stagnate.
    Heuristic,
}

#[derive(Debug, Clone, Copy)]
pub struct FeedbackConfig {
    pub mode: FeedbackTempMode,
    /// Softmax temperature of the fed-back expected token embedding
    /// softmax(logits / T) @ E_scaled; T -> 0 approaches the argmax token's
    /// embedding (i.e. normal generation), T -> inf the mean embedding.
    /// Used directly by Fixed and as the base value by Heuristic.
    pub temperature: f64,
    /// Target entropy (nats) of the feedback distribution: exact target for
    /// EntropyServo and OneSidedServo, band center for Heuristic.
    pub target_entropy: f64,
    /// Half-width of the entropy band for Heuristic.
    pub band: f64,
    /// Multiplicative decay/growth factor for Heuristic.
    pub decay: f64,
    /// Consecutive-input cosine above which Heuristic decays the temperature.
    pub max_cosine: f64,
    /// Weight of the expected embedding in the decode input
    /// (1 - a) * E[sampled] + a * expected; 1.0 is the pure expected input.
    pub blend: f64,
    /// Weight of the per-position expected embedding in the prefill input;
    /// 0 disables prefill feedback.
    pub prefill_blend: f64,
    /// Number of prefill-feedback refinement iterations (Schedule B
    /// cascade): iteration 0 observes the raw prompt, iteration k re-forwards
    /// `(1 - a_p) * E + a_p * expected(logits_{k-1})`; 1 is the single
    /// refinement pass, >= 2 the iterated cascade.
    pub prefill_iters: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FeedbackStats {
    pub temperature: f64,
    /// Entropy of the feedback distribution in nats.
    pub entropy: f64,
    /// Effective support exp(entropy).
    pub support: f64,
    /// L2 norm of the fed-back input (the blended input when blend < 1).
    pub input_norm: f64,
    /// Cosine between consecutive fed-back inputs; NaN on the first step.
    pub cosine_prev: f64,
}

/// Summary over all prefill positions of the prompt-feedback cascade.
#[derive(Debug, Clone)]
pub struct PrefillStats {
    /// Mean of the per-chunk temperatures of the observation pass.
    pub temperature: f64,
    /// Mean per-position entropy of the iteration-0 feedback distributions,
    /// nats.
    pub entropy: f64,
    /// Number of refinement iterations (total full forwards = iters + 1).
    pub iters: usize,
    /// Mean per-position entropy per iteration, nats (index 0 = iteration 0).
    pub entropies: Vec<f64>,
}

/// Entropy of softmax(logits / T) in nats.
fn entropy_at_temperature(logits: &Tensor, temperature: f64) -> Result<f64> {
    let probs = candle_nn::ops::softmax_last_dim(&(logits / temperature)?)?;
    let logp = (&probs + 1e-12)?.log()?;
    let h = (&probs * &logp)?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok(-h)
}

/// Solve T such that H(softmax(logits / T)) = target. H is non-decreasing
/// in T (dH/dT = Var(logits) / T^3), so a bisection over [1e-3, 20] applies;
/// targets outside the achievable range clamp to the bracket ends.
pub(crate) fn solve_entropy_temperature(logits: &Tensor, target: f64) -> Result<f64> {
    let (t_min, t_max) = (1e-3f64, 20.0f64);
    if entropy_at_temperature(logits, t_min)? >= target {
        return Ok(t_min);
    }
    if entropy_at_temperature(logits, t_max)? <= target {
        return Ok(t_max);
    }
    let mut lo = t_min;
    let mut hi = t_max;
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        if entropy_at_temperature(logits, mid)? < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    feedback: FeedbackConfig,
    /// Current temperature for the Heuristic mode; reset by clear_kv_cache.
    heuristic_t: f64,
    /// Previous fed-back input and its norm, for the consecutive cosine.
    last_input: Option<(Tensor, f64)>,
    /// Optional trained attention-pooling enricher; when present it replaces
    /// the softmax-blend formula for the decode input.
    pub enricher: Option<crate::enrich::Enricher>,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder, feedback: FeedbackConfig) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_full = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb.device(),
            None,
        )?);
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
            sliding_window: cfg.sliding_window,
            feedback,
            heuristic_t: feedback.temperature,
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
        if self.enricher.is_some() {
            let state = last.squeeze(1)?.squeeze(0)?.contiguous()?;
            if let Some(enr) = &mut self.enricher {
                enr.push_state(state);
            }
        }
        Ok(out)
    }

    /// Feed the previous step's expected token embedding back into the first
    /// block, skipping the embedding lookup of the sampled token. Also
    /// resolves the feedback temperature for the configured mode and returns
    /// the per-step feedback statistics.
    pub fn forward_feedback(
        &mut self,
        logits: &Tensor,
        sampled_token: u32,
        seqlen_offset: usize,
    ) -> Result<(Tensor, FeedbackStats)> {
        if self.enricher.is_some() {
            let window = self.enricher.as_ref().unwrap().window().to_vec();
            let token = Tensor::new(&[sampled_token], &self.device)?.unsqueeze(0)?;
            let e = self
                .embed_tokens
                .forward(&token)?
                .squeeze(0)?
                .squeeze(0)?;
            let input = self.enricher.as_ref().unwrap().enrich(&e, &window)?;
            let input = input.unsqueeze(0)?.unsqueeze(0)?;
            let (attention_mask, sliding_attention_mask) =
                self.create_attention_masks(1, 1, seqlen_offset)?;
            let xs = self.run_layers(input, &attention_mask, &sliding_attention_mask, seqlen_offset)?;
            let last = xs.narrow(1, 0, 1)?;
            let out = self.finish(last.clone())?;
            if let Some(enr) = &mut self.enricher {
                enr.push_state(last.squeeze(1)?.squeeze(0)?.contiguous()?);
            }
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

    fn feedback_input(&mut self, logits: &Tensor, sampled_token: u32) -> Result<(Tensor, FeedbackStats)> {
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
    /// expected embeddings from full-position logits, then `prefill_iters`
    /// refinement iterations re-forward the blended inputs
    /// `(1 - a) * E_prompt + a * expected(logits_{k-1})` over all positions
    /// with fresh caches; the last iteration's cache is the one the decode
    /// loop continues from. Returns last-position logits and summary stats.
    pub fn forward_prompt_blend(
        &mut self,
        input_ids: &Tensor,
    ) -> Result<(Tensor, PrefillStats)> {
        let (b_size, seq_len) = input_ids.dims2()?;
        const CHUNK: usize = 64;
        let alpha = self.feedback.prefill_blend.clamp(0.0, 1.0);
        let iters = self.feedback.prefill_iters.max(1);

        // Scaled prompt embeddings E * sqrt(h) in F32: the base input every
        // iteration blends against. (Gather + elementwise scaling, so
        // identical to embedding per chunk.)
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
            let chunk_h = -((&probs * &logp)?.sum_all()?.to_scalar::<f32>()? as f64)
                / (b_size * len) as f64;
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
                if self.enricher.is_some() {
                    let state = last.squeeze(1)?.squeeze(0)?.contiguous()?;
                    if let Some(enr) = &mut self.enricher {
                        enr.push_state(state);
                    }
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
    pub fn set_enricher(&mut self, enr: crate::enrich::Enricher) {
        self.enricher = Some(enr);
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Scaled input embeddings E[tokens] * sqrt(h) in F32, pad positions
    /// zeroed (for the training forward).
    pub fn embed_scale(&self, input_ids: &Tensor, lens: &[usize]) -> Result<Tensor> {
        let (b, seq) = input_ids.dims2()?;
        let e = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;
        let e = e.affine((self.hidden_size as f64).sqrt(), 0.0)?;
        let mask_data: Vec<f32> = (0..b)
            .flat_map(|r| {
                (0..seq).map(move |j| if j < lens[r] { 1.0f32 } else { 0.0 })
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (b, seq, 1), &self.device)?;
        Ok(e.broadcast_mul(&mask)?)
    }

    /// Raw (unscaled) embedding row of a single token, shape [hidden], F32.
    pub fn embed_one(&self, token: &Tensor) -> Result<Tensor> {
        Ok(self
            .embed_tokens
            .forward(token)?
            .to_dtype(DType::F32)?
            .squeeze(0)?)
    }

    /// Expected embedding probs @ E * sqrt(h) from a probability vector
    /// [1, vocab]; returns [1, hidden] in F32.
    pub fn expected_from_probs(&self, probs: &Tensor) -> Result<Tensor> {
        let xs = probs.matmul(self.embed_tokens.embeddings())?;
        Ok(xs
            .affine((self.hidden_size as f64).sqrt(), 0.0)?
            .to_dtype(DType::F32)?)
    }

    /// Cache-free, pad-aware training forward over caller-built input
    /// embeddings (batch, seq, hidden). Returns the final residual states
    /// for all positions and the logits for the answer range
    /// [answer_start, answer_start + answer_len).
    pub fn forward_inputs_train(
        &self,
        xs: &Tensor,
        lens: &[usize],
        answer_start: usize,
        answer_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b_size, seq_len, _) = xs.dims3()?;
        let full_mask = prepare_train_mask(b_size, seq_len, lens, None, self.dtype, &self.device)?;
        let slide_mask = prepare_train_mask(
            b_size,
            seq_len,
            lens,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        let mut h = xs.clone();
        for layer in self.layers.iter() {
            let mask = if layer.sliding_window.is_some() {
                &slide_mask
            } else {
                &full_mask
            };
            h = layer.forward_train(&h, mask)?;
        }
        let states = h.clone();
        let answer_states = h.narrow(1, answer_start, answer_len)?;
        let logits = self.finish(answer_states)?;
        Ok((states, logits))
    }

    pub fn clear_kv_cache(&mut self) {
        self.reset_layer_caches();
        self.heuristic_t = self.feedback.temperature;
        self.last_input = None;
        if let Some(enr) = &mut self.enricher {
            enr.clear_states();
        }
    }

    pub fn probe(&mut self, token: u32) -> Result<ProbeOut> {        let row = self
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
        let top5 = order.iter().take(5).map(|&i| (i as u32, logits[i])).collect();
        Ok(((row_v[..5].to_vec(), norm), top5))
    }
}
