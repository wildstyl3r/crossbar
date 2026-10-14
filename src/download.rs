//! On-demand download of the Gemma-3 weights from Hugging Face.
//!
//! A fresh clone has no `models/` directory; on the first run the binary
//! offers to fetch the quantized GGUF (default) or the safetensors backend
//! into the standard layout, so every existing `--model-dir` default keeps
//! working. The GGUF repo ships only the model file — the tokenizer and
//! chat template live in the (gated) google repo and its (non-gated)
//! unsloth mirror; we use google's exact files when an HF token is
//! available and fall back to the mirror otherwise.

use anyhow::{Context as _, Result};
use std::io::{IsTerminal, Read as _, Write as _};
use std::path::{Path, PathBuf};

use crate::cli::Args;

/// Quantized backend (the recommended default), from the non-gated
/// lmstudio mirror.
pub const DEFAULT_GGUF_DIR: &str = "models/gemma-3-270m-it-qat-GGUF";
const GGUF_REPO: &str = "lmstudio-community/gemma-3-270m-it-qat-GGUF";
const GGUF_FILE: &str = "gemma-3-270m-it-qat-Q4_0.gguf";

/// Where the safetensors backend and the shared tokenizer/template files
/// live (the `--fallback-dir` default).
pub const DEFAULT_SAFETENSORS_DIR: &str = "models/gemma-3-270m-it";
/// The gated upstream (exact files; needs license acceptance + a token).
const GOOGLE_REPO: &str = "google/gemma-3-270m-it";
/// The non-gated mirror (tokenizer/template slightly modified; used when
/// no HF token is available).
const UNSLOTH_REPO: &str = "unsloth/gemma-3-270m-it";

const HF_BASE: &str = "https://huggingface.co";

/// Files both backends need next to the weights.
const TOKENIZER_FILES: &[&str] =
    &["tokenizer.json", "chat_template.jinja", "generation_config.json"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Quantized GGUF (~241 MB, no HF account needed).
    Gguf,
    /// BF16 safetensors (~1.1 GB, gated: license acceptance + HF token).
    Safetensors,
}

/// One (repo, remote file, destination) download; an empty repo marks the
/// tokenizer-side files, whose source depends on token availability.
struct Wanted {
    repo: &'static str,
    file: &'static str,
    dest: PathBuf,
}

fn wanted_files(backend: Backend, fallback_dir: &Path) -> Vec<Wanted> {
    let mut files = match backend {
        Backend::Gguf => vec![Wanted {
            repo: GGUF_REPO,
            file: GGUF_FILE,
            dest: Path::new(DEFAULT_GGUF_DIR).join(GGUF_FILE),
        }],
        Backend::Safetensors => vec![
            Wanted {
                repo: GOOGLE_REPO,
                file: "model.safetensors",
                dest: fallback_dir.join("model.safetensors"),
            },
            Wanted {
                repo: GOOGLE_REPO,
                file: "config.json",
                dest: fallback_dir.join("config.json"),
            },
        ],
    };
    files.extend(TOKENIZER_FILES.iter().map(|name| Wanted {
        repo: "",
        file: name,
        dest: fallback_dir.join(name),
    }));
    files
}

/// HF token: `HF_TOKEN` env var, else the `huggingface-cli login` token
/// file. `None` => the gated google repo cannot be downloaded.
fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN")
        && !t.trim().is_empty()
    {
        return Some(t.trim().to_string());
    }
    let home = std::env::var("HOME").ok()?;
    read_token_line(&Path::new(&home).join(".cache/huggingface/token"))
}

fn read_token_line(path: &Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let line = s.lines().map(str::trim).find(|l| !l.is_empty())?;
    Some(line.to_string())
}

/// Which backend should the default `--model-dir` resolve to? Prefers the
/// quantized GGUF; falls back to safetensors when only that has weights.
fn resolved_backend(gguf_dir: &Path, safetensors_dir: &Path) -> Option<Backend> {
    if crate::assets::find_gguf_file(gguf_dir).is_some() {
        Some(Backend::Gguf)
    } else if crate::assets::find_weight_files(safetensors_dir).is_ok() {
        Some(Backend::Safetensors)
    } else {
        None
    }
}

/// Menu choice parsing: "" (Enter) and "1" pick the recommended GGUF.
fn parse_choice(line: &str) -> Option<Result<Backend>> {
    match line.trim() {
        "" | "1" => Some(Ok(Backend::Gguf)),
        "2" => Some(Ok(Backend::Safetensors)),
        "3" => Some(Err(anyhow::anyhow!("no model downloaded; exiting"))),
        _ => None,
    }
}

fn prompt_backend() -> Result<Backend> {
    let hint = "No model weights found under models/. \
        They can be downloaded now from Hugging Face:\n  \
        1) quantized GGUF  ~241 MB  lmstudio-community/gemma-3-270m-it-qat-GGUF  (recommended)\n  \
        2) safetensors     ~1.1 GB  google/gemma-3-270m-it  (requires accepting \
        Google's license at https://huggingface.co/google/gemma-3-270m-it and an HF token)\n  \
        3) exit\n\
        Choose [1]: ";
    loop {
        print!("{hint}");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            anyhow::bail!(
                "no input; run again with --download gguf (or safetensors) to download non-interactively"
            );
        }
        match parse_choice(&line) {
            Some(choice) => return choice,
            None => println!("please answer 1, 2 or 3"),
        }
    }
}

fn token_setup_instructions() -> anyhow::Error {
    anyhow::anyhow!(
        "the safetensors backend is hosted in the gated repo {GOOGLE_REPO}: \
         accept Google's Gemma license at https://huggingface.co/{GOOGLE_REPO} \
         (instant, needs a HF account), then expose a token via HF_TOKEN or \
         `huggingface-cli login`, and retry"
    )
}

/// Stream a remote file into `dest` (atomic: `<dest>.download` + rename),
/// resuming an interrupted attempt via HTTP Range and retrying a few times
/// on mid-body drops.
fn download_file(repo: &str, file: &str, dest: &Path, token: Option<&str>) -> Result<()> {
    const MAX_ATTEMPTS: usize = 3;
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        if attempt > 1 {
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        match download_attempt(repo, file, dest, token) {
            Ok(()) => return Ok(()),
            Err(e) => {
                if attempt < MAX_ATTEMPTS {
                    eprintln!("failed ({e:#}); retrying ({attempt}/{MAX_ATTEMPTS})");
                }
                last_err = Some(e);
            }
        }
    }
    Err(last_err.expect("at least one attempt"))
}

fn download_attempt(repo: &str, file: &str, dest: &Path, token: Option<&str>) -> Result<()> {
    let url = format!("{HF_BASE}/{repo}/resolve/main/{file}");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .build();
    let tmp = dest.with_extension("download");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    // Resume a previously interrupted attempt from the bytes already on
    // disk (the server may ignore the range: a 200 means start over).
    let have = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    let mut req = agent.get(&url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| match e {
        ureq::Error::Status(401 | 403, _) => anyhow::anyhow!(
            "{repo}/{file}: rejected (gated repo without an accepted license / valid token?)"
        ),
        ureq::Error::Status(code, _) => anyhow::anyhow!("{repo}/{file}: HTTP {code}"),
        e => anyhow::anyhow!("{repo}/{file}: {e}"),
    })?;
    if resp.status() == 416 {
        // The tmp file already covers the whole object; drop it and let a
        // fresh attempt re-download from zero.
        std::fs::remove_file(&tmp)?;
        anyhow::bail!("{repo}/{file}: stale partial download");
    }
    let resume = resp.status() == 206;
    let chunk = resp
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok());
    let total = chunk.map(|c| if resume { have + c } else { c });
    let mut out = std::io::BufWriter::new(if resume {
        // Append the remaining bytes behind what the last attempt wrote.
        std::fs::OpenOptions::new().append(true).create(true).open(&tmp)?
    } else {
        std::fs::File::create(&tmp)?
    });
    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut got = if resume { have } else { 0 };
    let interactive = std::io::stderr().is_terminal();
    if !interactive {
        eprint!(
            "downloading {repo}/{file}{}{}... ",
            total
                .map(|t| format!(" (~{:.0} MB)", t as f64 / 1e6))
                .unwrap_or_default(),
            if resume { " (resuming)" } else { "" }
        );
        let _ = std::io::stderr().flush();
    }
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        got += n as u64;
        if interactive
            && let Some(total) = total
        {
            eprint!(
                "\rdownloading {file}: {:.0}/{:.0} MB ({:.0}%)    ",
                got as f64 / 1e6,
                total as f64 / 1e6,
                100.0 * got as f64 / total.max(1) as f64
            );
        }
    }
    out.flush()?;
    drop(out);
    if interactive {
        eprintln!();
    } else {
        eprintln!("done");
    }
    // A connection that drops silently shows up as a short body: verify
    // against the announced length before renaming.
    if let Some(total) = total
        && std::fs::metadata(&tmp)?.len() < total
    {
        anyhow::bail!("{repo}/{file}: incomplete ({} of {total} bytes)", std::fs::metadata(&tmp)?.len());
    }
    std::fs::rename(&tmp, dest).with_context(|| format!("cannot finalize {}", dest.display()))?;
    Ok(())
}

/// Fetch everything `backend` needs that is not on disk yet. Tokenizer /
/// template files come from google when a token is available and from the
/// unsloth mirror otherwise. Returns whether anything was downloaded.
fn fetch_backend(backend: Backend, fallback_dir: &Path) -> Result<bool> {
    let token = hf_token();
    let mut fetched = false;
    for w in wanted_files(backend, fallback_dir) {
        if w.dest.exists() {
            continue;
        }
        let (repo, needs_token) = if w.repo.is_empty() {
            match &token {
                Some(_) => (GOOGLE_REPO, true),
                None => {
                    eprintln!(
                        "note: no HF token found — tokenizer/template files come from the \
                         non-gated mirror {UNSLOTH_REPO} (slightly modified vs {GOOGLE_REPO})"
                    );
                    (UNSLOTH_REPO, false)
                }
            }
        } else {
            (w.repo, w.repo == GOOGLE_REPO)
        };
        if needs_token && token.is_none() {
            return Err(token_setup_instructions());
        }
        let tok = (repo == GOOGLE_REPO).then_some(token.as_deref()).flatten();
        download_file(repo, w.file, &w.dest, tok)
            .with_context(|| format!("downloading {repo}/{}", w.file))?;
        fetched = true;
    }
    Ok(fetched)
}

/// Resolve the actual model/fallback dirs, applying the CLI defaults: the
/// default `--model-dir` prefers the GGUF dir and falls back to the
/// safetensors dir when only that has weights.
pub fn resolve_dirs(args: &Args) -> (PathBuf, PathBuf) {
    resolve_dirs_in(
        args,
        Path::new(DEFAULT_GGUF_DIR),
        Path::new(DEFAULT_SAFETENSORS_DIR),
    )
}

fn resolve_dirs_in(args: &Args, gguf_dir: &Path, st_dir: &Path) -> (PathBuf, PathBuf) {
    let model_dir = if args.model_dir == DEFAULT_GGUF_DIR {
        match resolved_backend(gguf_dir, st_dir) {
            Some(Backend::Safetensors) => st_dir.to_path_buf(),
            _ => gguf_dir.to_path_buf(),
        }
    } else {
        PathBuf::from(&args.model_dir)
    };
    let fallback_dir = if args.fallback_dir == DEFAULT_SAFETENSORS_DIR {
        st_dir.to_path_buf()
    } else {
        PathBuf::from(&args.fallback_dir)
    };
    (model_dir, fallback_dir)
}

/// Entry point, called at the top of `main` before any asset is loaded.
/// Never touches anything when the user overrode `--model-dir` /
/// `--fallback-dir`, or when the resolved default backend is complete.
pub fn ensure_weights(args: &Args) -> Result<()> {
    let gguf_dir = Path::new(DEFAULT_GGUF_DIR);
    let st_dir = Path::new(DEFAULT_SAFETENSORS_DIR);
    let explicit = args.model_dir != DEFAULT_GGUF_DIR || args.fallback_dir != DEFAULT_SAFETENSORS_DIR;
    if explicit {
        return Ok(());
    }
    let backend = match args.download {
        Some(b) => b,
        None => match resolved_backend(gguf_dir, st_dir) {
            // A backend is present; fetch anything only if its shared
            // tokenizer files are missing.
            Some(b) => {
                if TOKENIZER_FILES.iter().all(|f| st_dir.join(f).exists()) {
                    return Ok(());
                }
                b
            }
            None => {
                if std::io::stdin().is_terminal() {
                    prompt_backend()?
                } else {
                    anyhow::bail!(
                        "no model weights found and no terminal to prompt in: \
                         re-run with --download gguf (or --download safetensors), \
                         or place weights under {DEFAULT_GGUF_DIR} / {DEFAULT_SAFETENSORS_DIR} manually"
                    );
                }
            }
        },
    };
    if fetch_backend(backend, st_dir)? {
        println!("model weights ready");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_choice_defaults_to_gguf() {
        assert_eq!(parse_choice("").unwrap().unwrap(), Backend::Gguf);
        assert_eq!(parse_choice(" 1 ").unwrap().unwrap(), Backend::Gguf);
        assert_eq!(parse_choice("2").unwrap().unwrap(), Backend::Safetensors);
        assert!(parse_choice("3").unwrap().is_err());
        assert!(parse_choice("4").is_none());
        assert!(parse_choice("yes").is_none());
    }

    #[test]
    fn read_token_line_skips_blank_lines() {
        let dir = std::env::temp_dir().join(format!("crossbar-token-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("token");
        std::fs::write(&path, "\n  hf_abc123  \n\n").unwrap();
        assert_eq!(read_token_line(&path).as_deref(), Some("hf_abc123"));
        std::fs::write(&path, "\n\n").unwrap();
        assert_eq!(read_token_line(&path), None);
        assert_eq!(read_token_line(&dir.join("missing")), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wanted_files_cover_the_backend() {
        let fb = Path::new("models/gemma-3-270m-it");
        let gguf = wanted_files(Backend::Gguf, fb);
        assert_eq!(gguf.len(), 4); // gguf + 3 tokenizer files
        assert_eq!(gguf[0].file, GGUF_FILE);
        assert_eq!(gguf[0].repo, GGUF_REPO);
        assert!(gguf[1..].iter().all(|w| w.repo.is_empty()));

        let st = wanted_files(Backend::Safetensors, fb);
        assert_eq!(st.len(), 5); // weights + config + 3 tokenizer files
        assert!(st[..2].iter().all(|w| w.repo == GOOGLE_REPO));
    }

    #[test]
    fn resolve_dirs_prefers_the_backend_with_weights() {
        let mut args = <crate::cli::Args as clap::Parser>::parse_from(["crossbar"]);
        let gguf_dir = Path::new("models/gguf-test");
        let st_dir = Path::new("models/st-test");
        // Nothing anywhere: defaults resolve to the GGUF dir.
        let (m, f) = resolve_dirs_in(&args, gguf_dir, st_dir);
        assert_eq!(m, gguf_dir);
        assert_eq!(f, st_dir);
        // Explicit override passes through untouched.
        args.model_dir = "/somewhere/else".into();
        let (m, _) = resolve_dirs_in(&args, gguf_dir, st_dir);
        assert_eq!(m, PathBuf::from("/somewhere/else"));
    }
}
