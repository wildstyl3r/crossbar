use anyhow::{Context, Error as E, Result};
use candle::{DType, Device, Tensor};
use std::collections::HashSet;
use std::io::Write;
use tokenizers::Tokenizer;

use crate::Engine;
use crate::model::FeedbackStats;
use crate::sampling::{LogitsProcessor, apply_repeat_penalty};

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

pub struct GenerationOutcome {
    /// Full generated text (special tokens skipped).
    pub text: String,
    /// Number of generated tokens (including a trailing EOS).
    pub tokens: usize,
    /// Mean feedback-distribution entropy over the steps that used feedback.
    pub mean_hfb: Option<f64>,
    /// Mean raw-distribution entropy (T=1) over all steps.
    pub mean_raw_h: Option<f64>,
}

pub(crate) struct GenerationTiming {
    pub prompt_len: usize,
    pub prefill: std::time::Duration,
    pub decode_tokens: usize,
    pub decode_forward: std::time::Duration,
    pub decode_sample: std::time::Duration,
    pub decode_total: std::time::Duration,
    pub total: std::time::Duration,
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
    softmax_feedback: bool,
    prefill_feedback: bool,
    diagnostics: bool,
    collect_stats: bool,
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
        softmax_feedback: bool,
        prefill_feedback: bool,
        diagnostics: bool,
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
            softmax_feedback,
            prefill_feedback,
            diagnostics,
            collect_stats: false,
        }
    }

    /// Collect per-example mean entropies (used by the benchmark); does not
    /// enable the stderr diagnostics rows.
    pub fn set_collect_stats(&mut self, on: bool) {
        self.collect_stats = on;
    }

    /// Generation core. With `stream` the tokens are printed as they are
    /// produced; otherwise the run is silent and the text is returned.
    pub(crate) fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        stream: bool,
    ) -> Result<(GenerationOutcome, GenerationTiming)> {
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
        let has_enricher = self.model.has_enricher();
        let mut generated_tokens = 0usize;
        let mut prefill_total = std::time::Duration::ZERO;
        let mut decode_tokens = 0usize;
        let mut decode_forward = std::time::Duration::ZERO;
        let mut decode_sample = std::time::Duration::ZERO;
        let mut decode_total = std::time::Duration::ZERO;
        let start_gen = std::time::Instant::now();
        let mut prev_fb: Option<(Tensor, u32)> = None;
        let mut hfb_sum = 0f64;
        let mut hfb_n = 0usize;
        let mut raw_sum = 0f64;
        let mut raw_n = 0usize;
        for index in 0..sample_len {
            let step_start = std::time::Instant::now();
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let t = std::time::Instant::now();
            let (logits, fb_stats): (Tensor, Option<FeedbackStats>) = if has_enricher {
                // Enricher attached: the prompt is processed by the
                // sequential forced-token cascade (never the parallel
                // prefill) and every decode step is enriched from the
                // previous state.
                if index == 0 {
                    let mut last = None;
                    for (pos, &tok) in tokens.iter().enumerate() {
                        last = Some(self.model.cascade_step(tok, pos)?);
                    }
                    (last.expect("non-empty prompt"), None)
                } else {
                    let prev_token = tokens[tokens.len() - 1];
                    (self.model.cascade_step(prev_token, start_pos)?, None)
                }
            } else {
                match prev_fb.take() {
                    Some((prev_logits, prev_token)) => {
                        let (l, s) =
                            self.model
                                .forward_feedback(&prev_logits, prev_token, start_pos)?;
                        (l, Some(s))
                    }
                    None => {
                        let ctxt = &tokens[start_pos..];
                        let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
                        if index == 0 && self.prefill_feedback {
                            let (l, ps) = self.model.forward_prompt_blend(&input)?;
                            if self.diagnostics {
                                if ps.iters > 1 {
                                    let per_iter = ps
                                        .entropies
                                        .iter()
                                        .map(|h| format!("{h:.4}"))
                                        .collect::<Vec<_>>()
                                        .join(",");
                                    eprintln!(
                                        "# prefill-cascade: iters={}, T={:.4}, mean_H={:.4}, per_iter_H=[{per_iter}]",
                                        ps.iters, ps.temperature, ps.entropy
                                    );
                                } else {
                                    eprintln!(
                                        "# prefill-blend: T={:.4}, mean_H={:.4}",
                                        ps.temperature, ps.entropy
                                    );
                                }
                            }
                            (l, None)
                        } else {
                            (self.model.forward(&input, start_pos)?, None)
                        }
                    }
                }
            };
            let forward = t.elapsed();
            let mut fb_logits = if self.softmax_feedback {
                Some(logits.clone())
            } else {
                None
            };

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

            let want_stats = self.diagnostics || self.collect_stats;
            let diag_raw = if want_stats {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let log_z = max as f64 + (sum as f64).ln();
                let pl: f64 = exps
                    .iter()
                    .zip(logits.iter())
                    .map(|(e, l)| (e / sum) as f64 * (*l as f64))
                    .sum();
                Some((exps, sum, log_z - pl))
            } else {
                None
            };

            if self.repeat_penalty != 1. {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                apply_repeat_penalty(&mut logits, self.repeat_penalty, &tokens[start_at..]);
            }

            let next_token = self.logits_processor.sample(&mut logits);
            prev_fb = fb_logits.take().map(|l| (l, next_token));
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
            if let Some(s) = &fb_stats
                && !s.entropy.is_nan()
            {
                hfb_sum += s.entropy;
                hfb_n += 1;
            }
            if let Some((_, _, raw_h)) = &diag_raw {
                raw_sum += *raw_h;
                raw_n += 1;
            }
            if self.diagnostics {
                if index == 0 {
                    eprintln!("# step,T,H_fb,support,norm_x,cos_prev,raw_H,surprisal,distinct2");
                }
                let (exps, sum, raw_h) = diag_raw.as_ref().unwrap();
                let surprisal = (*sum as f64 / exps[next_token as usize] as f64).ln();
                let generated = &tokens[prompt_len..];
                let win = &generated[generated.len().saturating_sub(33)..];
                let mut bigrams: HashSet<(u32, u32)> = HashSet::new();
                let mut count = 0usize;
                for w in win.windows(2) {
                    bigrams.insert((w[0], w[1]));
                    count += 1;
                }
                let distinct2 = if count > 0 {
                    bigrams.len() as f64 / count as f64
                } else {
                    1.0
                };
                let (t_used, h_fb, support, norm_x, cos_prev) = match fb_stats {
                    Some(s) => (
                        s.temperature,
                        s.entropy,
                        s.support,
                        s.input_norm,
                        s.cosine_prev,
                    ),
                    None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
                };
                eprintln!(
                    "{},{:.4},{:.4},{:.2},{:.3},{:.4},{:.4},{:.4},{:.3}",
                    index, t_used, h_fb, support, norm_x, cos_prev, raw_h, surprisal, distinct2
                );
            }
            if self.eos_tokens.contains(&next_token) {
                break;
            }
            if stream && let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let text = self
            .tokenizer
            .tokenizer()
            .decode(&tokens[prompt_len..], true)
            .map_err(E::msg)
            .unwrap_or_default();
        let outcome = GenerationOutcome {
            text,
            tokens: generated_tokens,
            mean_hfb: (hfb_n > 0).then_some(hfb_sum / hfb_n as f64),
            mean_raw_h: (raw_n > 0).then_some(raw_sum / raw_n as f64),
        };
        let timing = GenerationTiming {
            prompt_len,
            prefill: prefill_total,
            decode_tokens,
            decode_forward,
            decode_sample,
            decode_total,
            total: start_gen.elapsed(),
        };
        Ok((outcome, timing))
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        let (outcome, timing) = self.generate(prompt, sample_len, true)?;
        if let Some(rest) = self.tokenizer.decode_rest()? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{} tokens generated ({:.2} token/s)",
            outcome.tokens,
            outcome.tokens as f64 / timing.total.as_secs_f64(),
        );
        println!(
            "prefill: {} tokens in {:.1} ms ({:.2} ms/token)",
            timing.prompt_len,
            timing.prefill.as_secs_f64() * 1000.0,
            timing.prefill.as_secs_f64() * 1000.0 / timing.prompt_len.max(1) as f64,
        );
        if timing.decode_tokens > 0 {
            let decode_ms = timing.decode_total.as_secs_f64() * 1000.0;
            let forward_ms = timing.decode_forward.as_secs_f64() * 1000.0;
            let sample_ms = timing.decode_sample.as_secs_f64() * 1000.0;
            let output_ms = (timing.decode_total - timing.decode_forward - timing.decode_sample)
                .as_secs_f64()
                * 1000.0;
            let pct = |ms: f64| 100.0 * ms / decode_ms.max(f64::MIN_POSITIVE);
            println!(
                "decode: {} tokens, avg {:.2} ms/token ({:.2} token/s) | \
                 model {:.2} ms ({:.1}%), sampling {:.2} ms ({:.1}%), output {:.2} ms ({:.1}%)",
                timing.decode_tokens,
                decode_ms / timing.decode_tokens as f64,
                timing.decode_tokens as f64 / timing.decode_total.as_secs_f64(),
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
