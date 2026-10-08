//! Model-asset discovery and generation-config parsing.

use anyhow::Result;

#[derive(serde::Deserialize, Debug, Default)]
pub struct GenerationConfig {
    pub do_sample: Option<bool>,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    eos_token_id: Option<OneOrMany>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
enum OneOrMany {
    One(u32),
    Many(Vec<u32>),
}

impl GenerationConfig {
    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(OneOrMany::One(t)) => vec![*t],
            Some(OneOrMany::Many(ts)) => ts.clone(),
            None => Vec::new(),
        }
    }
}

pub fn find_weight_files(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
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

pub fn find_gguf_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
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

pub fn resolve_file(
    model_dir: &std::path::Path,
    fallback_dir: &std::path::Path,
    name: &str,
) -> Result<std::path::PathBuf> {
    let primary = model_dir.join(name);
    if primary.exists() {
        return Ok(primary);
    }
    let fallback = fallback_dir.join(name);
    if fallback.exists() {
        return Ok(fallback);
    }
    anyhow::bail!(
        "neither {} nor {} found",
        primary.display(),
        fallback.display()
    )
}
