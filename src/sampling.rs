use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct LogitsProcessor {
    rng: StdRng,
    temperature: Option<f64>,
    top_k: Option<usize>,
    top_p: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(
        seed: u64,
        temperature: Option<f64>,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            temperature,
            top_k,
            top_p,
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let temperature = self.temperature.unwrap_or(1.0);
        if temperature < 1e-8 {
            return argmax(logits);
        }
        for l in logits.iter_mut() {
            *l /= temperature as f32;
        }

        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut indices: Vec<usize> = (0..probs.len()).collect();
        match self.top_k.filter(|&k| k > 0 && k < probs.len()) {
            Some(k) => {
                indices.select_nth_unstable_by(k - 1, |&i, &j| probs[j].total_cmp(&probs[i]));
                indices.truncate(k);
                indices.sort_unstable_by(|&i, &j| probs[j].total_cmp(&probs[i]));
            }
            None => {
                indices.sort_unstable_by(|&i, &j| probs[j].total_cmp(&probs[i]));
            }
        }
        if let Some(p) = self.top_p {
            let p = p.clamp(0.0, 1.0) as f32;
            let mut cumsum = 0f32;
            let mut cut = indices.len();
            for (rank, &i) in indices.iter().enumerate() {
                cumsum += probs[i];
                if cumsum >= p {
                    cut = rank + 1;
                    break;
                }
            }
            indices.truncate(cut);
        }

        let sum: f32 = indices.iter().map(|&i| probs[i]).sum();
        let u = self.rng.random::<f32>() * sum;
        let mut acc = 0f32;
        for &i in &indices {
            acc += probs[i];
            if u < acc {
                return i as u32;
            }
        }
        indices.last().map(|&i| i as u32).unwrap_or_else(|| argmax(logits))
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, l) in logits.iter().enumerate() {
        if *l > logits[best] {
            best = i;
        }
    }
    best as u32
}

pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let tokens: HashSet<u32> = context.iter().copied().collect();
    for (token, logit) in logits.iter_mut().enumerate() {
        if tokens.contains(&(token as u32)) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}
