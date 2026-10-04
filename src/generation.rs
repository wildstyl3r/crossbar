use anyhow::{Context, Error as E, Result};
use candle::{DType, Device, Tensor};
use std::io::Write;
use tokenizers::Tokenizer;

use crate::Engine;
use crate::sampling::{apply_repeat_penalty, LogitsProcessor};

pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(E::msg)
            .context("cannot decode")
    }

    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

pub struct TextGeneration {
    model: Engine,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_tokens: Vec<u32>,
    debug_logits: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Engine,
        tokenizer: Tokenizer,
        seed: u64,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        eos_tokens: Vec<u32>,
        device: &Device,
        debug_logits: bool,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temperature, top_k, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            eos_tokens,
            device: device.clone(),
            debug_logits,
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        self.tokenizer.clear();
        self.model.clear_kv_cache();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let prompt_len = tokens.len();
        let mut generated_tokens = 0usize;
        let mut prefill_total = std::time::Duration::ZERO;
        let mut decode_tokens = 0usize;
        let mut decode_forward = std::time::Duration::ZERO;
        let mut decode_sample = std::time::Duration::ZERO;
        let mut decode_total = std::time::Duration::ZERO;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let step_start = std::time::Instant::now();
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let t = std::time::Instant::now();
            let logits = self.model.forward(&input, start_pos)?;
            let forward = t.elapsed();

            let t = std::time::Instant::now();
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let mut logits = logits.to_vec1::<f32>()?;

            if self.debug_logits && index == 0 {
                let mut order: Vec<usize> = (0..logits.len()).collect();
                order.sort_by(|&i, &j| logits[j].total_cmp(&logits[i]));
                for &i in order.iter().take(5) {
                    let text = self
                        .tokenizer
                        .tokenizer()
                        .decode(&[i as u32], true)
                        .unwrap_or_default();
                    println!("  logit[{i}] = {:.4} {:?}", logits[i], text);
                }
            }

            if self.repeat_penalty != 1. {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                apply_repeat_penalty(&mut logits, self.repeat_penalty, &tokens[start_at..]);
            }

            let next_token = self.logits_processor.sample(&mut logits);
            let sampling = t.elapsed();
            if index == 0 {
                prefill_total = step_start.elapsed();
            } else {
                decode_tokens += 1;
                decode_forward += forward;
                decode_sample += sampling;
                decode_total += step_start.elapsed();
            }
            tokens.push(next_token);
            generated_tokens += 1;
            if self.eos_tokens.contains(&next_token) {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest()? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        println!(
            "prefill: {prompt_len} tokens in {:.1} ms ({:.2} ms/token)",
            prefill_total.as_secs_f64() * 1000.0,
            prefill_total.as_secs_f64() * 1000.0 / prompt_len.max(1) as f64,
        );
        if decode_tokens > 0 {
            let decode_ms = decode_total.as_secs_f64() * 1000.0;
            let forward_ms = decode_forward.as_secs_f64() * 1000.0;
            let sample_ms = decode_sample.as_secs_f64() * 1000.0;
            let output_ms = (decode_total - decode_forward - decode_sample).as_secs_f64() * 1000.0;
            let pct = |ms: f64| 100.0 * ms / decode_ms.max(f64::MIN_POSITIVE);
            println!(
                "decode: {decode_tokens} tokens, avg {:.2} ms/token ({:.2} token/s) | \
                 model {:.2} ms ({:.1}%), sampling {:.2} ms ({:.1}%), output {:.2} ms ({:.1}%)",
                decode_ms / decode_tokens as f64,
                decode_tokens as f64 / decode_total.as_secs_f64(),
                forward_ms,
                pct(forward_ms),
                sample_ms,
                pct(sample_ms),
                output_ms,
                pct(output_ms),
            );
        }
        Ok(())
    }
}
