//! Gemma-3 model configuration (shared by the safetensors and GGUF backends).

use candle_nn::Activation;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(layer_types: Option<Vec<LayerType>>, pattern: usize) -> Config {
        Config {
            attention_bias: false,
            head_dim: 256,
            hidden_activation: Activation::GeluPytorchTanh,
            hidden_size: 640,
            intermediate_size: 2048,
            num_attention_heads: 16,
            num_hidden_layers: 12,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            rope_local_base_freq: 1e4,
            vocab_size: 262_000,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: 256,
            sliding_window: 512,
            sliding_window_pattern: pattern,
            max_position_embeddings: 32_768,
            layer_types,
        }
    }

    #[test]
    fn pattern_fallback_every_sixth_layer_full() {
        let c = cfg(None, 6);
        assert!(c.is_sliding_layer(0));
        assert!(!c.is_sliding_layer(5));
        assert!(c.is_sliding_layer(6));
        assert!(!c.is_sliding_layer(11));
    }

    #[test]
    fn layer_types_override_with_pattern_fallback() {
        let c = cfg(
            Some(vec![LayerType::SlidingAttention, LayerType::FullAttention]),
            6,
        );
        assert!(c.is_sliding_layer(0));
        assert!(!c.is_sliding_layer(1));
        assert!(!c.is_sliding_layer(5));
    }
}
