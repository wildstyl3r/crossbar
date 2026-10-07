use std::io::Write as _;

use anyhow::{Context, Result};
use candle::Device;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;

use crate::chat;
use crate::generation::TextGeneration;
use crate::model::{FeedbackConfig, FeedbackTempMode};
use crate::Engine;

struct BenchMode {
    label: &'static str,
    feedback: Option<FeedbackConfig>,
    /// Decode-time expected-embedding feedback.
    decode: bool,
    /// Prefill (prompt) feedback.
    prefill: bool,
}

fn fb_config(
    mode: FeedbackTempMode,
    temperature: f64,
    target_entropy: f64,
    blend: f64,
    prefill_blend: f64,
    prefill_iters: usize,
) -> FeedbackConfig {
    FeedbackConfig {
        mode,
        temperature,
        target_entropy,
        band: 1.0,
        decay: 0.8,
        max_cosine: 0.999,
        blend,
        prefill_blend,
        prefill_iters,
    }
}

#[derive(Clone)]
enum Target {
    Digits(String),
    Num(i64),
    Text(&'static [&'static str]),
}

struct Example {
    task: &'static str,
    id: usize,
    prompt: String,
    target: Target,
    max_tokens: usize,
}

fn shuffle<T>(v: &mut [T], rng: &mut StdRng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
}

fn build_examples(n: usize) -> Vec<Example> {
    let mut examples = Vec::new();

    for &len in &[5usize, 10, 20] {
        let task: &'static str = match len {
            5 => "digits-5",
            10 => "digits-10",
            _ => "digits-20",
        };
        let mut rng = StdRng::seed_from_u64(1000 + len as u64);
        for id in 0..n {
            let digits: Vec<String> = (0..len).map(|_| rng.random_range(0..10).to_string()).collect();
            let target = digits.join("");
            let shown = digits.join(" ");
            examples.push(Example {
                task,
                id,
                prompt: format!("Repeat these digits in order: {shown}. Answer with only the digits."),
                target: Target::Digits(target),
                max_tokens: 2 * len + 12,
            });
        }
    }

    const NONSENSE: &[&str] = &[
        "dax", "wug", "blicket", "zorpie", "toma", "rugg", "fep", "kiki", "gorb", "pil", "nad",
        "mell", "lorp", "seng", "tob", "cootie",
    ];
    let mut rng = StdRng::seed_from_u64(4000);
    for id in 0..n {
        let mut pool = NONSENSE.to_vec();
        shuffle(&mut pool, &mut rng);
        let chosen: Vec<&str> = pool[..5].to_vec();
        let numbers: Vec<i64> = (0..5).map(|_| rng.random_range(1..100)).collect();
        let statements: Vec<String> = chosen
            .iter()
            .zip(&numbers)
            .map(|(w, n)| format!("{w} is {n}"))
            .collect();
        let query = chosen[4];
        let answer = numbers[4];
        examples.push(Example {
            task: "induce",
            id,
            prompt: format!(
                "{}.\nQuestion: {query} is?\nAnswer with only the number.",
                statements.join(". ")
            ),
            target: Target::Num(answer),
            max_tokens: 8,
        });
    }

    const TRIVIA: &[(&str, Target)] = &[
        ("What is the capital of France?", Target::Text(&["paris"])),
        ("What is the capital of Germany?", Target::Text(&["berlin"])),
        ("What is the capital of Italy?", Target::Text(&["rome", "roma"])),
        ("What is the capital of Spain?", Target::Text(&["madrid"])),
        ("What is the capital of Portugal?", Target::Text(&["lisbon"])),
        ("What is the capital of Japan?", Target::Text(&["tokyo"])),
        ("What is the capital of China?", Target::Text(&["beijing", "peking"])),
        ("What is the capital of Russia?", Target::Text(&["moscow"])),
        ("What is the capital of Greece?", Target::Text(&["athens"])),
        ("What is the capital of Poland?", Target::Text(&["warsaw"])),
        ("What is the capital of Sweden?", Target::Text(&["stockholm"])),
        ("What is the capital of Norway?", Target::Text(&["oslo"])),
        ("What is the capital of the Netherlands?", Target::Text(&["amsterdam"])),
        ("What is the capital of Austria?", Target::Text(&["vienna", "wien"])),
        ("What is the capital of Egypt?", Target::Text(&["cairo"])),
        ("What is the capital of Canada?", Target::Text(&["ottawa"])),
        ("What is the capital of Australia?", Target::Text(&["canberra"])),
        ("What is the capital of Brazil?", Target::Text(&["brasilia"])),
        ("What is the capital of Mexico?", Target::Text(&["mexico city"])),
        ("What is the capital of Switzerland?", Target::Text(&["bern", "berne"])),
        ("How many legs does a spider have?", Target::Num(8)),
        ("How many days are in a week?", Target::Num(7)),
        ("How many continents are there on Earth?", Target::Num(7)),
        ("How many sides does a triangle have?", Target::Num(3)),
        ("How many minutes are in an hour?", Target::Num(60)),
        ("How many hours are in a day?", Target::Num(24)),
        ("How many strings does a standard guitar have?", Target::Num(6)),
        (
            "How many players of one soccer team are on the field at the same time?",
            Target::Num(11),
        ),
        (
            "What is the largest planet in our solar system?",
            Target::Text(&["jupiter"]),
        ),
        (
            "Which planet is closest to the sun?",
            Target::Text(&["mercury"]),
        ),
        (
            "What color is the sky on a clear day?",
            Target::Text(&["blue"]),
        ),
        (
            "What color do you get by mixing red and white paint?",
            Target::Text(&["pink"]),
        ),
        (
            "What is the largest ocean on Earth?",
            Target::Text(&["pacific"]),
        ),
        (
            "What language do people speak in Brazil?",
            Target::Text(&["portuguese"]),
        ),
        ("What do bees make?", Target::Text(&["honey"])),
        (
            "What is the tallest animal in the world?",
            Target::Text(&["giraffe"]),
        ),
        (
            "What season comes after summer?",
            Target::Text(&["autumn", "fall"]),
        ),
        (
            "Who wrote the play Romeo and Juliet?",
            Target::Text(&["shakespeare"]),
        ),
        (
            "What is the hardest natural substance?",
            Target::Text(&["diamond"]),
        ),
        (
            "What gas do plants absorb from the air?",
            Target::Text(&["carbon dioxide", "co2"]),
        ),
    ];
    for (id, (question, target)) in TRIVIA.iter().take(n).enumerate() {
        examples.push(Example {
            task: "trivia",
            id,
            prompt: (*question).to_string(),
            target: target.clone(),
            max_tokens: 16,
        });
    }

    examples
}

fn first_nonempty_line(text: &str) -> &str {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .map(str::trim)
        .unwrap_or("")
}

fn last_int(s: &str) -> Option<i64> {
    let mut last = None;
    let mut cur: Option<i64> = None;
    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            cur = Some(cur.unwrap_or(0).saturating_mul(10).saturating_add(d as i64));
        } else if let Some(v) = cur.take() {
            last = Some(v);
        }
    }
    if let Some(v) = cur {
        last = Some(v);
    }
    last
}

fn normalize_words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn contains_answer(words: &[String], alt: &str) -> bool {
    let alt_words = normalize_words(alt);
    if alt_words.is_empty() || words.len() < alt_words.len() {
        return false;
    }
    words
        .windows(alt_words.len())
        .any(|w| w == alt_words.as_slice())
}

/// Returns (correct, extracted prediction for the CSV).
fn score(ex: &Example, text: &str) -> (bool, String) {
    let line = first_nonempty_line(text);
    match &ex.target {
        Target::Digits(t) => {
            let got = line.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
            (got == *t, got)
        }
        Target::Num(n) => {
            let got = last_int(line);
            (got == Some(*n), got.map(|v| v.to_string()).unwrap_or_default())
        }
        Target::Text(alts) => {
            let words = normalize_words(line);
            let hit = alts.iter().any(|a| contains_answer(&words, a));
            (hit, line.chars().take(60).collect())
        }
    }
}

fn target_str(target: &Target) -> String {
    match target {
        Target::Digits(s) => s.clone(),
        Target::Num(n) => n.to_string(),
        Target::Text(alts) => alts.join("|"),
    }
}

fn binom_coeff(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    let mut c = 1f64;
    for i in 0..k {
        c *= (n - i) as f64 / (i + 1) as f64;
    }
    c
}

/// Exact two-sided McNemar test p-value for paired counts:
/// n01 = baseline wrong / mode right, n10 = baseline right / mode wrong.
fn mcnemar_p(n01: usize, n10: usize) -> f64 {
    let n = n01 + n10;
    if n == 0 {
        return 1.0;
    }
    let k = n01.min(n10);
    let mut p = 0f64;
    for i in 0..=k {
        p += binom_coeff(n, i);
    }
    (p * 2f64.powi(-(n as i32)) * 2.0).min(1.0)
}

pub fn run_bench(
    make_engine: &dyn Fn(FeedbackConfig) -> Result<Engine>,
    tokenizer: Tokenizer,
    chat_template: &str,
    eos_tokens: Vec<u32>,
    device: &Device,
    n: usize,
    out_path: &str,
    enricher_path: Option<&str>,
    tasks_filter: Option<&str>,
    modes_filter: Option<&str>,
) -> Result<()> {
    let examples = {
        let mut examples = build_examples(n);
        if let Some(filter) = tasks_filter {
            let keep: std::collections::HashSet<&str> =
                filter.split(',').map(|s| s.trim()).collect();
            let unknown: Vec<&str> = keep
                .iter()
                .copied()
                .filter(|t| !examples.iter().any(|e| e.task == *t))
                .collect();
            if !unknown.is_empty() {
                anyhow::bail!("unknown bench task(s): {}", unknown.join(", "));
            }
            let total = examples.len();
            examples.retain(|e| keep.contains(e.task));
            println!(
                "bench: task filter '{filter}': {} of {total} examples",
                examples.len()
            );
        }
        examples
    };
    let mut modes = vec![
        BenchMode { label: "baseline", feedback: None, decode: false, prefill: false },
    ];
    if enricher_path.is_some() {
        // The enricher mode is the sequential cascade everywhere (prefill
        // and decode); the hand-set cascade modes do not compose with it.
        modes.push(BenchMode {
            label: "enricher",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 1.0, 0.0, 1)),
            decode: true,
            prefill: false,
        });
    } else {
        modes.push(BenchMode {
            label: "blend-d0.5",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 0.5, 0.0, 1)),
            decode: true,
            prefill: false,
        });
        modes.push(BenchMode {
            label: "prefill-p0.5",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 1.0, 0.5, 1)),
            decode: false,
            prefill: true,
        });
        modes.push(BenchMode {
            label: "both-0.5",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 0.5, 0.5, 1)),
            decode: true,
            prefill: true,
        });
        modes.push(BenchMode {
            label: "cascade-k2",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 1.0, 0.5, 2)),
            decode: false,
            prefill: true,
        });
        modes.push(BenchMode {
            label: "cascade-k3",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 1.0, 0.5, 3)),
            decode: false,
            prefill: true,
        });
        modes.push(BenchMode {
            label: "cascade-k2-a25",
            feedback: Some(fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 1.0, 0.25, 2)),
            decode: false,
            prefill: true,
        });
    }
    if let Some(filter) = modes_filter {
        let keep: std::collections::HashSet<&str> = filter.split(',').map(|s| s.trim()).collect();
        let unknown: Vec<&&str> = keep
            .iter()
            .filter(|m| !modes.iter().any(|mode| mode.label == **m))
            .collect();
        if !unknown.is_empty() {
            let names: Vec<String> = unknown.iter().map(|s| s.to_string()).collect();
            anyhow::bail!("unknown bench mode(s): {}", names.join(", "));
        }
        modes.retain(|mode| keep.contains(mode.label));
    }
    if !modes.iter().any(|m| m.label == "baseline") {
        anyhow::bail!("the mode filter must keep 'baseline' (McNemar reference)");
    }
    println!(
        "bench: {} examples x {} modes, greedy decoding, repeat-penalty 1.0",
        examples.len(),
        modes.len()
    );

    let mut csv = std::fs::File::create(out_path)
        .with_context(|| format!("cannot create {out_path}"))?;
    writeln!(
        csv,
        "task,mode,id,prompt,target,prediction,correct,gen_tokens,mean_hfb,mean_raw_h"
    )?;

    let mut results: Vec<Vec<bool>> = modes.iter().map(|_| vec![false; examples.len()]).collect();
    let mut lens: Vec<Vec<usize>> = modes.iter().map(|_| vec![0; examples.len()]).collect();

    for (mi, mode) in modes.iter().enumerate() {
        let start = std::time::Instant::now();
        let feedback = mode
            .feedback
            .unwrap_or_else(|| fb_config(FeedbackTempMode::Fixed, 1.0, 2.5, 1.0, 0.0, 1));
        let mut engine = (make_engine)(feedback)?;
        if mode.label.starts_with("enricher") {
            if let Some(path) = enricher_path {
                engine.set_enricher(path)?;
                println!("  enricher: loaded from {path}");
            }
        }
        let mut pipeline = TextGeneration::new(
            engine,
            tokenizer.clone(),
            0,
            Some(0.0),
            None,
            None,
            1.0,
            64,
            eos_tokens.clone(),
            device,
            false,
            mode.decode,
            mode.prefill,
            false,
        );
        pipeline.set_collect_stats(true);
        for (xi, ex) in examples.iter().enumerate() {
            let rendered = chat::render(chat_template, None, &ex.prompt)?;
            let (outcome, _) = pipeline.generate(&rendered, ex.max_tokens, false)?;
            let (correct, pred) = score(ex, &outcome.text);
            results[mi][xi] = correct;
            lens[mi][xi] = outcome.tokens;
            writeln!(
                csv,
                "{},{},{},{:?},{},{:?},{},{},{},{}",
                ex.task,
                mode.label,
                ex.id,
                ex.prompt,
                target_str(&ex.target),
                pred,
                correct as u8,
                outcome.tokens,
                outcome
                    .mean_hfb
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_default(),
                outcome
                    .mean_raw_h
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_default(),
            )?;
        }
        println!(
            "  mode {:>15}: done in {:.0}s",
            mode.label,
            start.elapsed().as_secs_f64()
        );
    }
    csv.flush()?;

    // Accuracy table with McNemar p-values vs baseline.
    let tasks: Vec<&'static str> = {
        let mut t: Vec<&'static str> = examples.iter().map(|e| e.task).collect();
        t.dedup();
        t
    };
    println!("\nexact-match accuracy (McNemar p vs baseline in parentheses)");
    let mut header = format!("{:<10}", "task");
    for mode in &modes {
        header.push_str(&format!(" {:>17}", mode.label));
    }
    println!("{header}");
    for task in &tasks {
        let idxs: Vec<usize> = examples
            .iter()
            .enumerate()
            .filter(|(_, e)| e.task == *task)
            .map(|(i, _)| i)
            .collect();
        let mut row = format!("{:<10}", task);
        for (mi, _) in modes.iter().enumerate() {
            let correct = idxs.iter().filter(|&&i| results[mi][i]).count();
            let cell = if mi == 0 {
                format!("{:.0}% (n={})", 100.0 * correct as f64 / idxs.len() as f64, idxs.len())
            } else {
                let n01 = idxs.iter().filter(|&&i| !results[0][i] && results[mi][i]).count();
                let n10 = idxs.iter().filter(|&&i| results[0][i] && !results[mi][i]).count();
                let p = mcnemar_p(n01, n10);
                format!(
                    "{:.0}% (p={:.3})",
                    100.0 * correct as f64 / idxs.len() as f64,
                    p
                )
            };
            row.push_str(&format!(" {:>17}", cell));
        }
        println!("{row}");
    }

    println!("\nmean generated tokens (early EOS / cap effects)");
    let mut header = format!("{:<10}", "task");
    for mode in &modes {
        header.push_str(&format!(" {:>17}", mode.label));
    }
    println!("{header}");
    for task in &tasks {
        let idxs: Vec<usize> = examples
            .iter()
            .enumerate()
            .filter(|(_, e)| e.task == *task)
            .map(|(i, _)| i)
            .collect();
        let mut row = format!("{:<10}", task);
        for (mi, _) in modes.iter().enumerate() {
            let mean = idxs.iter().map(|&i| lens[mi][i]).sum::<usize>() as f64 / idxs.len() as f64;
            row.push_str(&format!(" {:>17}", format!("{mean:.1}")));
        }
        println!("{row}");
    }
    println!("\nresults written to {out_path}");
    Ok(())
}
