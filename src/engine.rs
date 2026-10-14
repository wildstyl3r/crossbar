//! Backend dispatch: the safetensors model (F32 on CPU, BF16 on CUDA) vs
//! the GGUF (quantized) model.

use anyhow::Result;

use crate::experiments::cascade::CascadeTrainOut;
use crate::experiments::enrich;
use crate::experiments::feedback::{FeedbackStats, PrefillStats};
use crate::model;
use crate::quantized_model;

pub enum Engine {
    Safetensors(model::Model),
    Quantized(quantized_model::QuantizedModel),
}

impl Engine {
    pub fn forward(
        &mut self,
        input_ids: &candle::Tensor,
        pos: usize,
    ) -> candle::Result<candle::Tensor> {
        match self {
            Self::Safetensors(m) => m.forward(input_ids, pos),
            Self::Quantized(m) => m.forward(input_ids, pos),
        }
    }

    pub fn forward_feedback(
        &mut self,
        logits: &candle::Tensor,
        sampled_token: u32,
        pos: usize,
    ) -> candle::Result<(candle::Tensor, FeedbackStats)> {
        match self {
            Self::Safetensors(m) => m.forward_feedback(logits, sampled_token, pos),
            Self::Quantized(m) => m.forward_feedback(logits, sampled_token, pos),
        }
    }

    pub fn forward_prompt_blend(
        &mut self,
        input_ids: &candle::Tensor,
    ) -> candle::Result<(candle::Tensor, PrefillStats)> {
        match self {
            Self::Safetensors(m) => m.forward_prompt_blend(input_ids),
            Self::Quantized(m) => m.forward_prompt_blend(input_ids),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            Self::Safetensors(m) => m.clear_kv_cache(),
            Self::Quantized(m) => m.clear_kv_cache(),
        }
    }

    pub fn set_enricher(&mut self, path: &str) -> Result<()> {
        let enr = enrich::Enricher::load(path)?;
        match self {
            Self::Safetensors(m) => m.set_enricher(enr),
            Self::Quantized(m) => m.set_enricher(enr),
        }
        Ok(())
    }

    pub fn has_enricher(&self) -> bool {
        match self {
            Self::Safetensors(m) => m.has_enricher(),
            Self::Quantized(m) => m.has_enricher(),
        }
    }

    /// One sequential-cascade step (enriched input from the previous state,
    /// single-position forward). Used for the forced-token prefill and every
    /// decode step while the enricher is attached.
    pub fn cascade_step(&mut self, token: u32, pos: usize) -> candle::Result<candle::Tensor> {
        match self {
            Self::Safetensors(m) => m.cascade_step(token, pos),
            Self::Quantized(m) => m.cascade_step(token, pos),
        }
    }

    /// Sequential cascade training forward (TBPTT); safetensors backend only.
    #[allow(clippy::too_many_arguments)]
    pub fn cascade_train_forward(
        &self,
        enr: &enrich::Enricher,
        input_ids: &candle::Tensor,
        targets: &candle::Tensor,
        answer_start: usize,
        tau: usize,
        kv_grad: bool,
        forced_prob: f64,
        gt: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<CascadeTrainOut> {
        match self {
            Self::Safetensors(m) => Ok(m.cascade_train_forward(
                enr,
                input_ids,
                targets,
                answer_start,
                tau,
                kv_grad,
                forced_prob,
                gt,
                rng,
            )?),
            Self::Quantized(_) => anyhow::bail!("training requires the safetensors backend"),
        }
    }

    pub fn hidden(&self) -> usize {
        match self {
            Self::Safetensors(m) => m.hidden_size(),
            Self::Quantized(m) => m.hidden_size(),
        }
    }

    /// Maximum context window (positions) the backend supports.
    pub fn max_positions(&self) -> usize {
        match self {
            Self::Safetensors(m) => m.max_positions(),
            Self::Quantized(m) => m.max_positions(),
        }
    }

    /// Attention sliding window (0 when every layer is full attention).
    pub fn sliding_window(&self) -> usize {
        match self {
            Self::Safetensors(m) => m.sliding_window(),
            Self::Quantized(m) => m.sliding_window(),
        }
    }

    pub fn attach_enricher(&mut self, enr: enrich::Enricher) -> Result<()> {
        match self {
            Self::Safetensors(m) => m.set_enricher(enr),
            Self::Quantized(_) => anyhow::bail!("training requires the safetensors backend"),
        }
        Ok(())
    }
}
