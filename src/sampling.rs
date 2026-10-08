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
        indices
            .last()
            .map(|&i| i as u32)
            .unwrap_or_else(|| argmax(logits))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut p = LogitsProcessor::new(0, Some(0.0), None, None);
        let mut lg = vec![0.1f32, 2.0, -1.0, 0.5];
        assert_eq!(p.sample(&mut lg), 1);
    }

    #[test]
    fn top_k_one_picks_argmax() {
        let mut p = LogitsProcessor::new(42, Some(1.0), Some(1), None);
        let mut lg = vec![0.1f32, 2.0, -1.0, 0.5];
        assert_eq!(p.sample(&mut lg), 1);
    }

    #[test]
    fn sample_is_deterministic_per_seed() {
        let mut a = LogitsProcessor::new(7, Some(1.5), Some(3), Some(0.9));
        let mut b = LogitsProcessor::new(7, Some(1.5), Some(3), Some(0.9));
        let mut la = vec![0.1f32, 2.0, -1.0, 0.5, 1.7];
        let mut lb = la.clone();
        assert_eq!(a.sample(&mut la), b.sample(&mut lb));
    }

    #[test]
    fn repeat_penalty_downweights_context_tokens() {
        let mut lg = vec![1.0f32, -2.0, 0.5];
        apply_repeat_penalty(&mut lg, 2.0, &[1]);
        assert_eq!(lg[0], 1.0);
        assert!((lg[1] - (-4.0)).abs() < 1e-6);
        assert!((lg[2] - 0.5).abs() < 1e-6);
    }
}
