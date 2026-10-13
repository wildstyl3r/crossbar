use anyhow::{Context, Error as E, Result};
use candle::{DType, Device, Tensor};
use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use tokenizers::Tokenizer;

use crate::Engine;
use crate::experiments::feedback::FeedbackStats;
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

/// Per-request sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature; `None` means 1.0, `Some(0.0)` is greedy.
    pub temperature: Option<f64>,
    /// Top-k sampling; `None` or 0 disables it.
    pub top_k: Option<usize>,
    /// Nucleus sampling cutoff; `None` disables it.
    pub top_p: Option<f64>,
    /// Penalty applied to tokens in the recent context, 1. means none.
    pub repeat_penalty: f32,
    /// Context length for the repeat penalty.
    pub repeat_last_n: usize,
    /// RNG seed; `None` picks a fresh random seed per generation.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: Some(0.0),
            ..Default::default()
        }
    }

    fn processor(&self) -> LogitsProcessor {
        let seed = self.seed.unwrap_or_else(rand::random::<u64>);
        LogitsProcessor::new(seed, self.temperature, self.top_k, self.top_p)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// An EOS / `<end_of_turn>` token was sampled.
    Eos,
    /// The token budget was exhausted.
    MaxTokens,
    /// A stop string was about to be produced (the outcome text is
    /// truncated before it).
    Stopped(String),
    /// The cancel flag was set between decode steps.
    Cancelled,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eos => write!(f, "stop"),
            Self::MaxTokens => write!(f, "length"),
            Self::Stopped(s) => write!(f, "stop-string {s:?}"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

pub struct GenerationOutcome {
    /// Full generated text (special tokens skipped, stop strings removed).
    pub text: String,
    /// Number of generated tokens (including a trailing EOS).
    pub tokens: usize,
    /// Why generation stopped.
    pub reason: StopReason,
    /// Token ids present in the KV cache when generation ended: the ids
    /// forwarded by this and any reused prefix. Passing them as
    /// `GenerateOptions::cached_prefix` on a prompt that starts with them
    /// skips the re-prefill of that part.
    pub context_ids: Vec<u32>,
    /// Mean feedback-distribution entropy over the steps that used feedback.
    pub mean_hfb: Option<f64>,
    /// Mean raw-distribution entropy (T=1) over all steps.
    pub mean_raw_h: Option<f64>,
}

pub(crate) struct GenerationTiming {
    pub prompt_len: usize,
    /// Ids served from the retained KV cache instead of being re-prefilled.
    pub reused_prefix: usize,
    pub prefill: std::time::Duration,
    pub decode_tokens: usize,
    pub decode_forward: std::time::Duration,
    pub decode_sample: std::time::Duration,
    pub decode_total: std::time::Duration,
    pub total: std::time::Duration,
}

/// Logits filter hook (structured output / grammar constraints): `mask`
/// runs after the repeat penalty and before sampling; `on_token` receives
/// each sampled token so constrained-decoding state machines can advance.
pub trait LogitsFilter {
    fn mask(&mut self, logits: &mut [f32]);
    fn on_token(&mut self, _token: u32) {}
}

/// Streaming sink: receives each incrementally decoded chunk (and the
/// final withheld remainder).
pub type StreamSink<'a> = Box<dyn FnMut(&str) -> Result<()> + 'a>;

/// Per-generation options beyond the fixed engine setup.
pub struct GenerateOptions<'a> {
    /// Sampling parameters (temperature / top-k / top-p / penalties / seed).
    pub sampling: SamplingParams,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Generation stops (the stopping token is not streamed) when the
    /// decoded text would contain any of these strings.
    pub stop_strings: Vec<String>,
    /// Polled between decode steps; a `true` value stops generation.
    pub cancel: Option<&'a AtomicBool>,
    /// Ids already present in the KV cache from a previous generation
    /// (`GenerationOutcome::context_ids`); when they prefix this prompt,
    /// that part is not re-prefilled. Ignored for the enricher and
    /// feedback modes, and on any mismatch (the cache is cleared).
    pub cached_prefix: Option<Vec<u32>>,
    /// Optional logits filter (schema-constrained sampling).
    pub filter: Option<Box<dyn LogitsFilter + 'a>>,
    /// Streaming sink; `None` runs silently.
    pub stream: Option<StreamSink<'a>>,
}

impl Default for GenerateOptions<'_> {
    fn default() -> Self {
        Self {
            sampling: SamplingParams::default(),
            max_tokens: 256,
            stop_strings: Vec::new(),
            cancel: None,
            cached_prefix: None,
            filter: None,
            stream: None,
        }
    }
}

/// Number of leading ids of `full` that match `cached` — the part of the
/// prompt that can be served from the retained KV cache. 0 when there is
/// no cached prefix or it does not match.
fn resume_len(full: &[u32], cached: Option<&[u32]>) -> usize {
    let Some(cached) = cached else {
        return 0;
    };
    if cached.len() > full.len() {
        return 0;
    }
    if full[..cached.len()] == *cached {
        cached.len()
    } else {
        0
    }
}

pub struct TextGeneration {
    model: Engine,
    device: Device,
    tokenizer: TokenOutputStream,
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
        eos_tokens: Vec<u32>,
        device: &Device,
        debug_logits: bool,
        softmax_feedback: bool,
        prefill_feedback: bool,
        diagnostics: bool,
    ) -> Self {
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
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

    /// Access to the underlying tokenizer (e.g. to build a vocab trie).
    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer.tokenizer()
    }

    /// Maximum context window (positions) of the underlying model.
    pub fn max_context(&self) -> usize {
        self.model.max_positions()
    }

    /// Generation core. With a `stream` sink the tokens are handed to it as
    /// they are produced; otherwise the run is silent and the text is
    /// returned in the outcome.
    pub(crate) fn generate(
        &mut self,
        prompt: &str,
        opts: GenerateOptions<'_>,
    ) -> Result<(GenerationOutcome, GenerationTiming)> {
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        self.generate_from(tokens, opts)
    }

    /// Generation core over pre-built prompt ids (used by the REPL to
    /// assemble append-only prompts from the retained KV-cache prefix).
    pub(crate) fn generate_from(
        &mut self,
        mut tokens: Vec<u32>,
        mut opts: GenerateOptions<'_>,
    ) -> Result<(GenerationOutcome, GenerationTiming)> {
        self.tokenizer.clear();
        let mut logits_processor = opts.sampling.processor();

        let prompt_len = tokens.len();
        let has_enricher = self.model.has_enricher();
        // KV-cache reuse: keep the cache when the caller's prefix (the ids
        // forwarded by the previous generation) is a prefix of this prompt
        // and the model runs the plain path — the feedback and enricher
        // modes carry cross-position state that does not survive a
        // partial prefill.
        let matched = resume_len(&tokens, opts.cached_prefix.as_deref());
        // A full-length match (no new tokens) would re-forward an already
        // cached position; treat it as no reuse.
        let can_reuse = matched > 0
            && matched < prompt_len
            && !has_enricher
            && !self.prefill_feedback
            && !self.softmax_feedback;
        let mut base = if can_reuse {
            matched
        } else {
            self.model.clear_kv_cache();
            0
        };
        let mut cache_len = if can_reuse { matched } else { 0 };
        let mut reused_prefix = 0usize;
        // A multi-token chunk append pairs correctly with the positional
        // masks only while the sliding ring has not wrapped (and from an
        // empty cache, where the >= window case returns the chunk in
        // positional order); feed the delta token by token otherwise.
        let chunked = base == 0 || prompt_len <= self.model.sliding_window();
        let mut prefill_total = std::time::Duration::ZERO;
        if can_reuse && opts.max_tokens > 0 {
            reused_prefix = base;
            if !chunked {
                let t = std::time::Instant::now();
                for (i, &tok) in tokens[base..prompt_len - 1].iter().enumerate() {
                    let input = Tensor::new(&[tok], &self.device)?.unsqueeze(0)?;
                    self.model.forward(&input, base + i)?;
                    cache_len += 1;
                }
                prefill_total += t.elapsed();
                base = prompt_len - 1;
            }
        }
        let mut generated_tokens = 0usize;
        let mut reason = StopReason::MaxTokens;
        let mut recent = String::new();
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
        for index in 0..opts.max_tokens {
            if let Some(cancel) = opts.cancel
                && cancel.load(Ordering::Relaxed)
            {
                reason = StopReason::Cancelled;
                break;
            }
            let step_start = std::time::Instant::now();
            let context_size = if index > 0 { 1 } else { tokens.len() - base };
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
            // the cache grew by the forwarded chunk (prefill) or token
            cache_len += context_size;
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

            if opts.sampling.repeat_penalty != 1. {
                let start_at = tokens.len().saturating_sub(opts.sampling.repeat_last_n);
                apply_repeat_penalty(
                    &mut logits,
                    opts.sampling.repeat_penalty,
                    &tokens[start_at..],
                );
            }

            if let Some(filter) = opts.filter.as_mut() {
                filter.mask(&mut logits);
            }

            let next_token = logits_processor.sample(&mut logits);
            if let Some(filter) = opts.filter.as_mut() {
                filter.on_token(next_token);
            }
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
                reason = StopReason::Eos;
                break;
            }
            if !opts.stop_strings.is_empty() {
                if let Ok(piece) = self.tokenizer.tokenizer().decode(&[next_token], true) {
                    recent.push_str(&piece);
                    let keep = 128usize;
                    if recent.chars().count() > keep {
                        let skip = recent.chars().count() - keep;
                        recent = recent.chars().skip(skip).collect();
                    }
                }
                if let Some(s) = opts
                    .stop_strings
                    .iter()
                    .find(|s| recent.contains(s.as_str()))
                {
                    reason = StopReason::Stopped(s.clone());
                    break;
                }
            }
            if let Some(sink) = opts.stream.as_mut()
                && let Some(t) = self.tokenizer.next_token(next_token)?
            {
                sink(&t)?;
            }
        }
        if let Some(sink) = opts.stream.as_mut()
            && let Some(rest) = self.tokenizer.decode_rest()?
        {
            sink(&rest)?;
        }
        let mut text = self
            .tokenizer
            .tokenizer()
            .decode(&tokens[prompt_len..], true)
            .map_err(E::msg)
            .unwrap_or_default();
        if let StopReason::Stopped(s) = &reason
            && let Some(pos) = text.find(s.as_str())
        {
            text.truncate(pos);
        }
        let outcome = GenerationOutcome {
            text,
            tokens: generated_tokens,
            reason,
            context_ids: tokens[..cache_len.min(tokens.len())].to_vec(),
            mean_hfb: (hfb_n > 0).then_some(hfb_sum / hfb_n as f64),
            mean_raw_h: (raw_n > 0).then_some(raw_sum / raw_n as f64),
        };
        let timing = GenerationTiming {
            prompt_len,
            reused_prefix,
            prefill: prefill_total,
            decode_tokens,
            decode_forward,
            decode_sample,
            decode_total,
            total: start_gen.elapsed(),
        };
        Ok((outcome, timing))
    }

    /// One-shot generation with stdout streaming plus the timing report
    /// (the `--prompt` CLI path).
    pub fn run_with(&mut self, prompt: &str, mut opts: GenerateOptions<'_>) -> Result<()> {
        if opts.stream.is_none() {
            opts.stream = Some(Box::new(|chunk: &str| -> Result<()> {
                print!("{chunk}");
                std::io::stdout().flush()?;
                Ok(())
            }));
        }
        let (outcome, timing) = self.generate(prompt, opts)?;
        std::io::stdout().flush()?;
        println!(
            "\n{} tokens generated ({:.2} token/s, reason: {})",
            outcome.tokens,
            outcome.tokens as f64 / timing.total.as_secs_f64(),
            outcome.reason,
        );
        println!(
            "prefill: {} tokens in {:.1} ms ({:.2} ms/token)",
            timing.prompt_len,
            timing.prefill.as_secs_f64() * 1000.0,
            timing.prefill.as_secs_f64() * 1000.0 / timing.prompt_len.max(1) as f64,
        );
        let prefilled = timing.prompt_len.saturating_sub(timing.reused_prefix);
        if timing.reused_prefix > 0 {
            println!(
                "kv reuse: {} tokens served from the cache, {} re-prefilled",
                timing.reused_prefix, prefilled
            );
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_len_requires_prefix_match() {
        let full = [10, 20, 30, 40, 50];
        assert_eq!(resume_len(&full, None), 0);
        assert_eq!(resume_len(&full, Some(&[10, 20, 30])), 3);
        assert_eq!(resume_len(&full, Some(&[10, 20, 30, 40, 50])), 5);
        assert_eq!(resume_len(&full, Some(&[10, 21, 30])), 0);
        assert_eq!(resume_len(&full, Some(&[11, 20])), 0);
        assert_eq!(resume_len(&full, Some(&[10, 20, 30, 40, 50, 60])), 0);
        assert_eq!(resume_len(&[], Some(&[1])), 0);
    }
}
