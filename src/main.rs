use anyhow::{Context, Error as E, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use tokenizers::Tokenizer;

use crossbar::assets::{GenerationConfig, find_gguf_file, resolve_file};
use crossbar::experiments::bench;
use crossbar::chat;
use crossbar::cli::Args;
use crossbar::config;
use crossbar::engine::Engine;
use crossbar::experiments::feedback::{FeedbackConfig, FeedbackTempMode};
use crossbar::generation;
use crossbar::model;
use crossbar::quantized_model;
use crossbar::repl;
use crossbar::experiments::train;

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
    println!(
        "backend: {}",
        if gguf_file.is_some() {
            "gguf (quantized)"
        } else {
            "safetensors"
        }
    );
    let make_engine = |feedback: FeedbackConfig| -> Result<Engine> {
        if let Some(ref gguf_path) = gguf_file {
            let mut reader = std::fs::File::open(gguf_path)?;
            Ok(Engine::Quantized(
                quantized_model::QuantizedModel::from_gguf(&mut reader, &device, feedback)?,
            ))
        } else {
            let config: config::Config = serde_json::from_reader(std::fs::File::open(
                resolve_file(model_dir, fallback_dir, "config.json")?,
            )?)?;
            let dtype = if device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            let weight_files = crossbar::assets::find_weight_files(model_dir)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
            Ok(Engine::F32(model::Model::new(&config, vb, feedback)?))
        }
    };
    let feedback = args.feedback_config();
    let mut engine = make_engine(feedback)?;
    if let Some(path) = &args.enricher {
        engine.set_enricher(path)?;
        println!("enricher: loaded from {path} (cascade prefill + decode)");
    }
    println!("loaded the model in {:?}", start.elapsed());

    let chat_template = std::fs::read_to_string(resolve_file(
        model_dir,
        fallback_dir,
        "chat_template.jinja",
    )?)?;

    if args.train_args.train_enricher {
        let spec = args.enricher_spec();
        if let Err(e) = spec.validate(engine.hidden()) {
            anyhow::bail!("invalid enricher spec: {e}");
        }
        if args.train_args.train_probe {
            train::run_probe(
                &engine,
                &tokenizer,
                &chat_template,
                &device,
                &args.train_args.train_out,
            )?;
            return Ok(());
        }
        let t = &args.train_args;
        train::run(
            &mut engine,
            &tokenizer,
            &chat_template,
            &device,
            t.train_steps_warmup,
            t.train_steps_st,
            &t.train_out,
            train::TrainConfig {
                gate_bias: t.train_gate_bias,
                lr_warmup: t.train_lr_warmup,
                gate_floor: t.train_gate_floor,
                gt_start: t.train_gt_start,
                gt_end: t.train_gt_end,
                inj_l2: t.train_inj_l2,
                tau: t.train_tau,
                kv_grad: t.train_kv_grad,
                batch: t.train_batch,
                accum: t.train_accum,
                sweep: t.train_sweep.clone(),
                grad_clip: t.train_grad_clip,
                seed: t.train_seed,
                skip_eval: t.train_skip_eval,
                spec,
            },
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
            && !eos_tokens.contains(&t)
        {
            eos_tokens.push(t);
        }
    }

    let fb = &args.feedback_args;
    if args.bench_args.bench {
        bench::run_bench(
            &make_engine,
            tokenizer,
            &chat_template,
            eos_tokens,
            &device,
            args.bench_args.bench_n,
            &args.bench_args.bench_out,
            args.enricher.as_deref(),
            args.bench_args.bench_tasks.as_deref(),
            args.bench_args.bench_modes.as_deref(),
        )?;
        return Ok(());
    }

    let do_sample = generation_config.do_sample.unwrap_or(true);
    let temperature = if do_sample {
        Some(
            args.temperature
                .or(generation_config.temperature)
                .unwrap_or(1.0),
        )
    } else {
        Some(0.0)
    };
    let top_k = args.top_k.or(generation_config.top_k);
    let top_p = args.top_p.or(generation_config.top_p);

    let sampling = generation::SamplingParams {
        temperature,
        top_k,
        top_p,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        seed: Some(args.seed),
    };

    if args.repl {
        let pipeline = generation::TextGeneration::new(
            engine,
            tokenizer,
            eos_tokens.clone(),
            &device,
            args.debug_logits,
            fb.softmax_feedback,
            fb.prefill_feedback_blend > 0.0,
            fb.softmax_feedback_diagnostics,
        );
        let config = repl::ReplConfig {
            sampling,
            max_tokens: 512,
            tool_mode: args.tools,
            system: args.system.clone(),
            max_positions: pipeline.max_context(),
        };
        repl::run(pipeline, chat_template, eos_tokens, config)?;
        return Ok(());
    }

    let prompt = args
        .prompt
        .as_deref()
        .context("--prompt is required unless --probe, --bench or --repl is set")?;
    let prompt = chat::render(&chat_template, args.system.as_deref(), prompt)?;

    println!(
        "temp: {:.2} top-k: {} top-p: {} repeat-penalty: {:.2} repeat-last-n: {}",
        temperature.unwrap_or(1.0),
        top_k.map(|k| k.to_string()).unwrap_or_else(|| "-".into()),
        top_p
            .map(|p| format!("{p:.2}"))
            .unwrap_or_else(|| "-".into()),
        args.repeat_penalty,
        args.repeat_last_n,
    );
    if fb.softmax_feedback {
        println!(
            "softmax feedback: on, mode: {:?}, temperature: {:.3}, target entropy: {:.2}, blend: {:.2}",
            fb.softmax_feedback_mode,
            fb.softmax_feedback_temperature,
            fb.softmax_feedback_target_entropy,
            fb.softmax_feedback_blend
        );
        if fb.softmax_feedback_mode == FeedbackTempMode::Heuristic {
            println!(
                "  heuristic band: {:.2}, decay: {:.2}, max cosine: {:.3}",
                fb.softmax_feedback_band, fb.softmax_feedback_decay, fb.softmax_feedback_max_cosine
            );
        }
        if fb.softmax_feedback_diagnostics {
            println!("  diagnostics: per-step CSV on stderr");
        }
    }
    if fb.prefill_feedback_blend > 0.0 {
        println!(
            "prefill feedback: on, blend: {:.2}",
            fb.prefill_feedback_blend
        );
        if fb.prefill_feedback_iters != 1 {
            println!("prefill cascade: iters: {}", fb.prefill_feedback_iters);
        }
    }

    let mut pipeline = generation::TextGeneration::new(
        engine,
        tokenizer,
        eos_tokens,
        &device,
        args.debug_logits,
        fb.softmax_feedback,
        fb.prefill_feedback_blend > 0.0,
        fb.softmax_feedback_diagnostics,
    );
    let opts = generation::GenerateOptions {
        sampling,
        max_tokens: args.sample_len,
        ..Default::default()
    };
    pipeline.run_with(&prompt, opts)?;
    Ok(())
}
