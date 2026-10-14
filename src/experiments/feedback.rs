//! Softmax-feedback configuration, statistics and entropy-solver math.

use candle::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum FeedbackTempMode {
    /// Constant temperature from --softmax-feedback-temperature.
    Fixed,
    /// Per-step servo: solve T so that H(softmax(logits / T)) equals the
    /// target entropy.
    EntropyServo,
    /// Like EntropyServo, but the solution is capped at T <= 1: the servo
    /// only tames entropy explosions (T < 1) and otherwise falls back to the
    /// model's own distribution (T = 1) instead of blurring confident steps.
    OneSidedServo,
    /// Persistent temperature, multiplicatively decayed/grown when the
    /// measured feedback statistics leave the target entropy band or the
    /// consecutive inputs stagnate.
    Heuristic,
}

#[derive(Debug, Clone, Copy)]
pub struct FeedbackConfig {
    pub mode: FeedbackTempMode,
    /// Softmax temperature of the fed-back expected token embedding
    /// softmax(logits / T) @ E_scaled; T -> 0 approaches the argmax token's
    /// embedding (i.e. normal generation), T -> inf the mean embedding.
    /// Used directly by Fixed and as the base value by Heuristic.
    pub temperature: f64,
    /// Target entropy (nats) of the feedback distribution: exact target for
    /// EntropyServo and OneSidedServo, band center for Heuristic.
    pub target_entropy: f64,
    /// Half-width of the entropy band for Heuristic.
    pub band: f64,
    /// Multiplicative decay/growth factor for Heuristic.
    pub decay: f64,
    /// Consecutive-input cosine above which Heuristic decays the temperature.
    pub max_cosine: f64,
    /// Weight of the expected embedding in the decode input
    /// (1 - a) * E[sampled] + a * expected; 1.0 is the pure expected input.
    pub blend: f64,
    /// Weight of the per-position expected embedding in the prefill input;
    /// 0 disables prefill feedback.
    pub prefill_blend: f64,
    /// Number of prefill-feedback refinement iterations (the iterated
    /// prefill cascade): iteration 0 observes the raw prompt, iteration k
    /// re-forwards `(1 - a_p) * E + a_p * expected(logits_{k-1})`; 1 is the
    /// single refinement pass, >= 2 the iterated cascade.
    pub prefill_iters: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FeedbackStats {
    pub temperature: f64,
    /// Entropy of the feedback distribution in nats.
    pub entropy: f64,
    /// Effective support exp(entropy).
    pub support: f64,
    /// L2 norm of the fed-back input (the blended input when blend < 1).
    pub input_norm: f64,
    /// Cosine between consecutive fed-back inputs; NaN on the first step.
    pub cosine_prev: f64,
}

/// Summary over all prefill positions of the prompt-feedback cascade.
#[derive(Debug, Clone)]
pub struct PrefillStats {
    /// Mean of the per-chunk temperatures of the observation pass.
    pub temperature: f64,
    /// Mean per-position entropy of the iteration-0 feedback distributions,
    /// nats.
    pub entropy: f64,
    /// Number of refinement iterations (total full forwards = iters + 1).
    pub iters: usize,
    /// Mean per-position entropy per iteration, nats (index 0 = iteration 0).
    pub entropies: Vec<f64>,
}

/// Entropy of softmax(logits / T) in nats.
fn entropy_at_temperature(logits: &Tensor, temperature: f64) -> Result<f64> {
    let probs = candle_nn::ops::softmax_last_dim(&(logits / temperature)?)?;
    let logp = (&probs + 1e-12)?.log()?;
    let h = (&probs * &logp)?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok(-h)
}

/// Solve T such that H(softmax(logits / T)) = target. H is non-decreasing
/// in T (dH/dT = Var(logits) / T^3), so a bisection over [1e-3, 20] applies;
/// targets outside the achievable range clamp to the bracket ends.
pub(crate) fn solve_entropy_temperature(logits: &Tensor, target: f64) -> Result<f64> {
    let (t_min, t_max) = (1e-3f64, 20.0f64);
    if entropy_at_temperature(logits, t_min)? >= target {
        return Ok(t_min);
    }
    if entropy_at_temperature(logits, t_max)? <= target {
        return Ok(t_max);
    }
    let mut lo = t_min;
    let mut hi = t_max;
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        if entropy_at_temperature(logits, mid)? < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn entropy_solver_recovers_temperature() {
        // Two-token logits [ln 3, 0]: H(softmax(logits)) = -(0.75 ln 0.75 + 0.25 ln 0.25).
        let lg = Tensor::new(&[[3f32.ln(), 0f32]], &Device::Cpu).unwrap();
        let target = -(0.75 * 0.75f64.ln() + 0.25 * 0.25f64.ln());
        let t = solve_entropy_temperature(&lg, target).unwrap();
        assert!((t - 1.0).abs() < 0.05, "t = {t}");
    }

    #[test]
    fn entropy_solver_clamps_out_of_range_targets() {
        let lg = Tensor::new(&[[3f32.ln(), 0f32]], &Device::Cpu).unwrap();
        assert_eq!(solve_entropy_temperature(&lg, 0.0).unwrap(), 1e-3);
        assert_eq!(solve_entropy_temperature(&lg, 100.0).unwrap(), 20.0);
    }

    #[test]
    fn entropy_is_monotone_in_temperature() {
        let lg = Tensor::new(&[[2f32, 0f32, -1f32]], &Device::Cpu).unwrap();
        let h1 = entropy_at_temperature(&lg, 0.5).unwrap();
        let h2 = entropy_at_temperature(&lg, 2.0).unwrap();
        assert!(h1 < h2);
    }
}
