mod bench;
mod chat;
mod enrich;
mod generation;
mod model;
mod quantized_model;
mod sampling;
mod train;

use anyhow::{Context, Error as E, Result};
use clap::Parser;
use candle::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The prompt to run the generator with (required unless --bench or
    /// --probe is set).
    #[arg(long, short = 'p')]
    prompt: Option<String>,

    /// Run the feedback-impact benchmark suite instead of generating from a
    /// prompt: greedy decoding over synthetic tasks (digit copy, arithmetic,
    /// counting, induction, trivia) across baseline and feedback modes.
    #[arg(long)]
    bench: bool,

    /// Number of examples per task for --bench (trivia caps at 40).
    #[arg(long, default_value_t = 100)]
    bench_n: usize,

    /// Output CSV path for --bench with per-example rows.
    #[arg(long, default_value = "bench_results.csv")]
    bench_out: String,

    /// Comma-separated task filter for --bench (digits-5, digits-10,
    /// digits-20, induce, trivia); default runs all tasks.
    #[arg(long)]
    bench_tasks: Option<String>,

    /// Comma-separated mode filter for --bench (baseline, blend-d0.5,
    /// prefill-p0.5, both-0.5, cascade-k2, cascade-k3, cascade-k2-a25);
    /// 'baseline' is always required. Default runs all modes.
    #[arg(long)]
    bench_modes: Option<String>,

    /// Optional system prompt, folded into the first user turn.
    #[arg(long)]
    system: Option<String>,

    /// Directory containing the model files; auto-detects GGUF (quantized) vs safetensors.
    #[arg(long, default_value = "models/gemma-3-270m-it")]
    model_dir: String,

    /// Directory to fall back to for tokenizer/chat template/generation config
    /// when they are missing from --model-dir.
    #[arg(long, default_value = "models/gemma-3-270m-it")]
    fallback_dir: String,

    /// The temperature used to generate samples, 0 means greedy decoding.
    #[arg(long)]
    temperature: Option<f64>,

    /// Top-k sampling, 0 disables it.
    #[arg(long)]
    top_k: Option<usize>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Print the top-5 logits of the first generated token (debug).
    #[arg(long)]
    debug_logits: bool,

    /// Dump embedding row and single-token logits for a token (debug).
    #[arg(long)]
    probe: bool,

    /// Feed the previous step's expected token embedding softmax(logits / T)
    /// @ E back into the first block during decoding, instead of embedding
    /// the sampled token; textual decoding stays normal.
    #[arg(long)]
    softmax_feedback: bool,

    /// Softmax temperature for --softmax-feedback; 0 approaches the argmax
    /// token's embedding (i.e. normal generation), large values approach the
    /// mean embedding. Used as the base value by the heuristic mode.
    #[arg(long, default_value_t = 1.0)]
    softmax_feedback_temperature: f64,

    /// Weight of the expected embedding in the decode input
    /// (1 - a) * E[sampled] + a * expected; 1.0 is the pure expected input.
    #[arg(long, default_value_t = 1.0)]
    softmax_feedback_blend: f64,

    /// Weight of the per-position expected embedding in the prefill input
    /// (prompt feedback, two-pass); 0 disables it.
    #[arg(long, default_value_t = 0.0)]
    prefill_feedback_blend: f64,

    /// Number of prefill-feedback refinement iterations (Schedule B
    /// cascade): iteration 0 observes the raw prompt, iteration k re-forwards
    /// (1 - a_p) * E + a_p * expected(logits_{k-1}) over all positions; 1 is
    /// the single refinement pass (the original prefill feedback), >= 2 the
    /// iterated cascade.
    #[arg(long, default_value_t = 1)]
    prefill_feedback_iters: usize,

    /// How the feedback temperature is chosen each step: the fixed value,
    /// an entropy-matched servo, or a threshold-based heuristic.
    #[arg(long, value_enum, default_value = "fixed")]
    softmax_feedback_mode: model::FeedbackTempMode,

    /// Target entropy (nats) of the feedback distribution for
    /// --softmax-feedback-mode entropy-servo (exact) and heuristic (band
    /// center).
    #[arg(long, default_value_t = 2.5)]
    softmax_feedback_target_entropy: f64,

    /// Half-width of the entropy band for --softmax-feedback-mode heuristic.
    #[arg(long, default_value_t = 1.0)]
    softmax_feedback_band: f64,

    /// Multiplicative decay/growth factor for --softmax-feedback-mode
    /// heuristic.
    #[arg(long, default_value_t = 0.8)]
    softmax_feedback_decay: f64,

    /// Consecutive-feedback cosine above which --softmax-feedback-mode
    /// heuristic decays the temperature.
    #[arg(long, default_value_t = 0.999)]
    softmax_feedback_max_cosine: f64,

    /// Print per-step feedback diagnostics as CSV lines to stderr.
    #[arg(long)]
    softmax_feedback_diagnostics: bool,

    /// Load a trained attention-pooling enricher checkpoint; replaces the
    /// softmax-blend decode feedback with the learned enrichment.
    #[arg(long)]
    enricher: Option<String>,

    /// Run the enricher training loop instead of generating from a prompt.
    #[arg(long)]
    train_enricher: bool,

    /// Teacher-forced CE steps before the straight-through phase.
    #[arg(long, default_value_t = 150)]
    train_steps_warmup: usize,

    /// Straight-through Gumbel free-running steps after warmup.
    #[arg(long, default_value_t = 300)]
    train_steps_st: usize,

    /// Where the trained enricher checkpoint is written / reloaded from.
    #[arg(long, default_value = "enricher.safetensors")]
    train_out: String,
}

enum Engine {
    F32(model::Model),
    Quantized(quantized_model::QuantizedModel),
}

impl Engine {
    fn forward(
        &mut self,
        input_ids: &candle::Tensor,
        pos: usize,
    ) -> candle::Result<candle::Tensor> {
        match self {
            Self::F32(m) => m.forward(input_ids, pos),
            Self::Quantized(m) => m.forward(input_ids, pos),
        }
    }

    fn forward_feedback(
        &mut self,
        logits: &candle::Tensor,
        sampled_token: u32,
        pos: usize,
    ) -> candle::Result<(candle::Tensor, model::FeedbackStats)> {
        match self {
            Self::F32(m) => m.forward_feedback(logits, sampled_token, pos),
            Self::Quantized(m) => m.forward_feedback(logits, sampled_token, pos),
        }
    }

    fn forward_prompt_blend(
        &mut self,
        input_ids: &candle::Tensor,
    ) -> candle::Result<(candle::Tensor, model::PrefillStats)> {
        match self {
            Self::F32(m) => m.forward_prompt_blend(input_ids),
            Self::Quantized(m) => m.forward_prompt_blend(input_ids),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::F32(m) => m.clear_kv_cache(),
            Self::Quantized(m) => m.clear_kv_cache(),
        }
    }

    fn probe(&mut self, token: u32) -> candle::Result<model::ProbeOut> {
        match self {
            Self::F32(m) => m.probe(token),
            Self::Quantized(m) => m.probe(token),
        }
    }

    fn set_enricher(&mut self, path: &str, r: usize, k: usize) -> Result<()> {
        let hidden = match self {
            Self::F32(m) => m.hidden_size(),
            Self::Quantized(m) => m.hidden_size(),
        };
        let enr = enrich::Enricher::load(path, hidden, r, k)?;
        match self {
            Self::F32(m) => m.set_enricher(enr),
            Self::Quantized(m) => m.set_enricher(enr),
        }
        Ok(())
    }

    fn forward_inputs_train(
        &self,
        xs: &candle::Tensor,
        lens: &[usize],
        answer_start: usize,
        answer_len: usize,
    ) -> candle::Result<(candle::Tensor, candle::Tensor)> {
        match self {
            Self::F32(m) => m.forward_inputs_train(xs, lens, answer_start, answer_len),
            Self::Quantized(_) => candle::bail!("training requires the safetensors backend"),
        }
    }

    fn embed_scale(
        &self,
        input_ids: &candle::Tensor,
        lens: &[usize],
    ) -> candle::Result<candle::Tensor> {
        match self {
            Self::F32(m) => m.embed_scale(input_ids, lens),
            Self::Quantized(_) => candle::bail!("training requires the safetensors backend"),
        }
    }

    fn embed_one(&self, token: &candle::Tensor) -> candle::Result<candle::Tensor> {
        match self {
            Self::F32(m) => m.embed_one(token),
            Self::Quantized(_) => candle::bail!("training requires the safetensors backend"),
        }
    }

    fn expected_from_probs(&self, probs: &candle::Tensor) -> candle::Result<candle::Tensor> {
        match self {
            Self::F32(m) => m.expected_from_probs(probs),
            Self::Quantized(_) => candle::bail!("training requires the safetensors backend"),
        }
    }

    fn hidden(&self) -> usize {
        match self {
            Self::F32(m) => m.hidden_size(),
            Self::Quantized(m) => m.hidden_size(),
        }
    }

    fn attach_enricher(&mut self, enr: enrich::Enricher) -> Result<()> {
        match self {
            Self::F32(m) => m.set_enricher(enr),
            Self::Quantized(_) => anyhow::bail!("training requires the safetensors backend"),
        }
        Ok(())
    }
}

#[derive(serde::Deserialize, Debug, Default)]
struct GenerationConfig {
    do_sample: Option<bool>,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
    eos_token_id: Option<OneOrMany>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

impl GenerationConfig {
    fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(OneOrMany::One(t)) => vec![*t],
            Some(OneOrMany::Many(ts)) => ts.clone(),
            None => Vec::new(),
        }
    }
}

fn find_weight_files(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        anyhow::bail!("no .safetensors files found in {}", dir.display());
    }
    Ok(files)
}

fn find_gguf_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut files: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    files.into_iter().next()
}

fn resolve_file(model_dir: &std::path::Path, fallback_dir: &std::path::Path, name: &str) -> Result<std::path::PathBuf> {
    let primary = model_dir.join(name);
    if primary.exists() {
        return Ok(primary);
    }
    let fallback = fallback_dir.join(name);
    if fallback.exists() {
        return Ok(fallback);
    }
    anyhow::bail!("neither {} nor {} found", primary.display(), fallback.display())
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let model_dir = std::path::Path::new(&args.model_dir);
    let fallback_dir = std::path::Path::new(&args.fallback_dir);
    let generation_config: GenerationConfig = match std::fs::File::open(resolve_file(
        model_dir,
        fallback_dir,
        "generation_config.json",
    )?) {
        Ok(file) => serde_json::from_reader(file)?,
        Err(_) => GenerationConfig::default(),
    };
    let tokenizer = Tokenizer::from_file(resolve_file(model_dir, fallback_dir, "tokenizer.json")?)
        .map_err(E::msg)?;
    let gguf_file = find_gguf_file(model_dir);
    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("backend: {}", if gguf_file.is_some() { "gguf (quantized)" } else { "safetensors" });
    let make_engine = |feedback: model::FeedbackConfig| -> Result<Engine> {
        if let Some(ref gguf_path) = gguf_file {
            let mut reader = std::fs::File::open(gguf_path)?;
            Ok(Engine::Quantized(quantized_model::QuantizedModel::from_gguf(
                &mut reader,
                &device,
                feedback,
            )?))
        } else {
            let config: model::Config = serde_json::from_reader(std::fs::File::open(resolve_file(
                model_dir,
                fallback_dir,
                "config.json",
            )?)?)?;
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let weight_files = find_weight_files(model_dir)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
            Ok(Engine::F32(model::Model::new(&config, vb, feedback)?))
        }
    };
    let feedback = model::FeedbackConfig {
        mode: args.softmax_feedback_mode,
        temperature: args.softmax_feedback_temperature,
        target_entropy: args.softmax_feedback_target_entropy,
        band: args.softmax_feedback_band,
        decay: args.softmax_feedback_decay,
        max_cosine: args.softmax_feedback_max_cosine,
        blend: args.softmax_feedback_blend,
        prefill_blend: args.prefill_feedback_blend,
        prefill_iters: args.prefill_feedback_iters,
    };
    let mut engine = make_engine(feedback)?;
    if let Some(path) = &args.enricher {
        engine.set_enricher(path, 192, 8)?;
        println!("enricher: loaded from {path}");
    }
    println!("loaded the model in {:?}", start.elapsed());

    let chat_template =
        std::fs::read_to_string(resolve_file(model_dir, fallback_dir, "chat_template.jinja")?)?;

    if args.train_enricher {
        train::run(
            &mut engine,
            &tokenizer,
            &chat_template,
            &device,
            args.train_steps_warmup,
            args.train_steps_st,
            &args.train_out,
        )?;
        return Ok(());
    }

    if args.probe {
        let token = tokenizer.token_to_id("Paris").context("no Paris token")?;
        let (row, top5) = engine.probe(token)?;
        println!("probe token {token}: row[:5]={row:?}");
        println!("probe token {token}: row norm={:.4}", row.1);
        for (i, l) in top5 {
            println!("  logit[{i}] = {l:.4}");
        }
        if let Some(gguf_path) = &gguf_file {
            quantized_model::probe_weights(&gguf_path.to_string_lossy(), &device)?;
        }
        return Ok(());
    }

    let mut eos_tokens = generation_config.eos_ids();
    for special in ["<eos>", "<end_of_turn>"] {
        if let Some(t) = tokenizer.token_to_id(special)
            && !eos_tokens.contains(&t) {
                eos_tokens.push(t);
            }
    }

    if args.bench {
        bench::run_bench(
            &make_engine,
            tokenizer,
            &chat_template,
            eos_tokens,
            &device,
            args.bench_n,
            &args.bench_out,
            args.enricher.as_deref(),
            args.bench_tasks.as_deref(),
            args.bench_modes.as_deref(),
        )?;
        return Ok(());
    }

    let prompt = args
        .prompt
        .as_deref()
        .context("--prompt is required unless --probe or --bench is set")?;
    let prompt = chat::render(&chat_template, args.system.as_deref(), prompt)?;

    let do_sample = generation_config.do_sample.unwrap_or(true);
    let temperature = if do_sample {
        Some(args.temperature.or(generation_config.temperature).unwrap_or(1.0))
    } else {
        Some(0.0)
    };
    let top_k = args.top_k.or(generation_config.top_k);
    let top_p = args.top_p.or(generation_config.top_p);
    println!(
        "temp: {:.2} top-k: {} top-p: {} repeat-penalty: {:.2} repeat-last-n: {}",
        temperature.unwrap_or(1.0),
        top_k.map(|k| k.to_string()).unwrap_or_else(|| "-".into()),
        top_p.map(|p| format!("{p:.2}")).unwrap_or_else(|| "-".into()),
        args.repeat_penalty,
        args.repeat_last_n,
    );
    if args.softmax_feedback {
        println!(
            "softmax feedback: on, mode: {:?}, temperature: {:.3}, target entropy: {:.2}, blend: {:.2}",
            args.softmax_feedback_mode,
            args.softmax_feedback_temperature,
            args.softmax_feedback_target_entropy,
            args.softmax_feedback_blend
        );
        if args.softmax_feedback_mode == model::FeedbackTempMode::Heuristic {
            println!(
                "  heuristic band: {:.2}, decay: {:.2}, max cosine: {:.3}",
                args.softmax_feedback_band,
                args.softmax_feedback_decay,
                args.softmax_feedback_max_cosine
            );
        }
        if args.softmax_feedback_diagnostics {
            println!("  diagnostics: per-step CSV on stderr");
        }
    }
    if args.prefill_feedback_blend > 0.0 {
        println!(
            "prefill feedback: on, blend: {:.2}",
            args.prefill_feedback_blend
        );
        if args.prefill_feedback_iters != 1 {
            println!("prefill cascade: iters: {}", args.prefill_feedback_iters);
        }
    }

    let mut pipeline = generation::TextGeneration::new(
        engine,
        tokenizer,
        args.seed,
        temperature,
        top_k,
        top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        eos_tokens,
        &device,
        args.debug_logits,
        args.softmax_feedback,
        args.prefill_feedback_blend > 0.0,
        args.softmax_feedback_diagnostics,
    );
    pipeline.run(&prompt, args.sample_len)?;
    Ok(())
}
