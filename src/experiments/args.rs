//! CLI argument groups for the experimental entry points, plus the
//! CLI-to-domain converters. Flattened into `cli::Args` by the binary; the
//! core CLI never grows when an experiment gains a knob.

use crate::cli::Args;
use crate::experiments::enrich;
use crate::experiments::feedback::{FeedbackConfig, FeedbackTempMode};
use crate::experiments::train;

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Run the feedback-impact benchmark suite instead of generating from a
    /// prompt: greedy decoding over synthetic tasks (digit copy, arithmetic,
    /// counting, induction, trivia) across baseline and feedback modes.
    #[arg(long)]
    pub bench: bool,

    /// Number of examples per task for --bench (trivia caps at 40).
    #[arg(long, default_value_t = 100)]
    pub bench_n: usize,

    /// Output CSV path for --bench with per-example rows.
    #[arg(long, default_value = "bench_results.csv")]
    pub bench_out: String,

    /// Comma-separated task filter for --bench (digits-5, digits-10,
    /// digits-20, induce, induce-sh, induce-early, trivia); default runs all
    /// tasks.
    #[arg(long)]
    pub bench_tasks: Option<String>,

    /// Comma-separated mode filter for --bench (baseline, blend-d0.5,
    /// prefill-p0.5, both-0.5, cascade-k2, cascade-k3, cascade-k2-a25);
    /// 'baseline' is always required. Default runs all modes.
    #[arg(long)]
    pub bench_modes: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct FeedbackArgs {
    /// Feed the previous step's expected token embedding softmax(logits / T)
    /// @ E back into the first block during decoding, instead of embedding
    /// the sampled token; textual decoding stays normal.
    #[arg(long)]
    pub softmax_feedback: bool,

    /// Softmax temperature for --softmax-feedback; 0 approaches the argmax
    /// token's embedding (i.e. normal generation), large values approach the
    /// mean embedding. Used as the base value by the heuristic mode.
    #[arg(long, default_value_t = 1.0)]
    pub softmax_feedback_temperature: f64,

    /// Weight of the expected embedding in the decode input
    /// (1 - a) * E[sampled] + a * expected; 1.0 is the pure expected input.
    #[arg(long, default_value_t = 1.0)]
    pub softmax_feedback_blend: f64,

    /// Weight of the per-position expected embedding in the prefill input
    /// (prompt feedback, two-pass); 0 disables it.
    #[arg(long, default_value_t = 0.0)]
    pub prefill_feedback_blend: f64,

    /// Number of prefill-feedback refinement iterations (the iterated
    /// prefill cascade): iteration 0 observes the raw prompt, iteration k
    /// re-forwards (1 - a_p) * E + a_p * expected(logits_{k-1}) over all
    /// positions; 1 is the single refinement pass (the original prefill
    /// feedback), >= 2 the iterated cascade.
    #[arg(long, default_value_t = 1)]
    pub prefill_feedback_iters: usize,

    /// How the feedback temperature is chosen each step: the fixed value,
    /// an entropy-matched servo, or a threshold-based heuristic.
    #[arg(long, value_enum, default_value = "fixed")]
    pub softmax_feedback_mode: FeedbackTempMode,

    /// Target entropy (nats) of the feedback distribution for
    /// --softmax-feedback-mode entropy-servo (exact) and heuristic (band
    /// center).
    #[arg(long, default_value_t = 2.5)]
    pub softmax_feedback_target_entropy: f64,

    /// Half-width of the entropy band for --softmax-feedback-mode heuristic.
    #[arg(long, default_value_t = 1.0)]
    pub softmax_feedback_band: f64,

    /// Multiplicative decay/growth factor for --softmax-feedback-mode
    /// heuristic.
    #[arg(long, default_value_t = 0.8)]
    pub softmax_feedback_decay: f64,

    /// Consecutive-feedback cosine above which --softmax-feedback-mode
    /// heuristic decays the temperature.
    #[arg(long, default_value_t = 0.999)]
    pub softmax_feedback_max_cosine: f64,

    /// Print per-step feedback diagnostics as CSV lines to stderr.
    #[arg(long)]
    pub softmax_feedback_diagnostics: bool,
}

#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Run the enricher training loop instead of generating from a prompt.
    #[arg(long)]
    pub train_enricher: bool,

    /// Teacher-forced CE steps before the hard-sampled ST phase.
    #[arg(long, default_value_t = 150)]
    pub train_steps_warmup: usize,

    /// Hard-sampled ST steps after warmup (scheduled forcing 1 -> 0).
    #[arg(long, default_value_t = 300)]
    pub train_steps_st: usize,

    /// Where the trained enricher checkpoint is written / reloaded from.
    #[arg(long, default_value = "enricher.safetensors")]
    pub train_out: String,

    /// Initial gate bias of a fresh enricher (+2.0 => sigmoid ~0.88: the
    /// gate starts open and must learn to close).
    #[arg(long, default_value_t = 2.0)]
    pub train_gate_bias: f64,

    /// Linear lr ramp length (steps) at the start of warmup (0 = constant).
    #[arg(long, default_value_t = 50)]
    pub train_lr_warmup: usize,

    /// Gate floor during warmup. The ST phase anneals 0.6 * this -> 0.
    #[arg(long, default_value_t = 0.5)]
    pub train_gate_floor: f64,

    /// Starting sampling temperature of the ST phase (annealed to
    /// --train-gt-end).
    #[arg(long, default_value_t = 1.0)]
    pub train_gt_start: f64,

    /// Anneal target of the ST sampling temperature (0.1 ~= the greedy
    /// deployment regime).
    #[arg(long, default_value_t = 0.1)]
    pub train_gt_end: f64,

    /// Relative pre-gate injection penalty weight: loss + lambda * mean(
    /// ||W shat||^2 / ||q||^2 ). 0 disables.
    #[arg(long, default_value_t = 0.1)]
    pub train_inj_l2: f64,

    /// BPTT truncation horizon (in-graph cascade steps at the sequence end).
    /// Measured (ENRICHER.md): the state-chain dies by ~4 steps (c^4 <= 6e-3
    /// at healthy injection); the exact-KV path is the dominant cross-step
    /// credit at init. tau=6 is the knee of the gradient/memory curve:
    /// 72% of tau=8's gradient at 78% of its memory (5.3 vs 6.8 GB peak).
    #[arg(long, default_value_t = 6)]
    pub train_tau: usize,

    /// Keep the full KV-path gradient in the training graph (default, exact:
    /// at W=0 the KV path is the ONLY cross-step credit; detaching it
    /// captured just 32% of the gradient in the tau-sweep). Disabling it
    /// detaches keys/values for a smaller graph at a large gradient cost.
    #[arg(long, default_value_t = true)]
    pub train_kv_grad: bool,

    /// Lockstep batch size per micro-step (4 halves the in-graph memory vs
    /// 8; the cascade forward is weight-streaming-bound, so batch 4 costs
    /// nearly the same wall time as 8).
    #[arg(long, default_value_t = 4)]
    pub train_batch: usize,

    /// Micro-steps accumulated per optimizer step: effective batch =
    /// batch x accum. batch 4 x accum 4 (effective 16) is the default
    /// recipe: ~3.5-4 GB peak, ~50 s/step; warmup 150 + ST 300 micros
    /// ~ 1.6-1.9 h, and a full night is ~3k micros ~ 7-10 h. accum 8 halves
    /// the step count budget; accum 16 only if noise demonstrably dominates.
    #[arg(long, default_value_t = 4)]
    pub train_accum: usize,

    /// Override the tau-sweep diagnostic grid ("tau:kv,tau:kv", e.g.
    /// "8:1"); empty runs the default comparison grid.
    #[arg(long, default_value = "")]
    pub train_sweep: String,

    /// Global-norm gradient clip over the enricher parameters.
    #[arg(long, default_value_t = 1.0)]
    pub train_grad_clip: f64,

    /// Seed for the training-data shuffle and the ST sampling rng.
    #[arg(long, default_value_t = 31337)]
    pub train_seed: u64,

    /// Skip the terminal evaluation (smoke runs).
    #[arg(long)]
    pub train_skip_eval: bool,

    /// Probe a trained enricher checkpoint (--train-out) for the ST
    /// soft-mixture artifact instead of training.
    #[arg(long)]
    pub train_probe: bool,

    /// Enricher readout rank: 0 = v0 full-rank W (default, unchanged), d > 0
    /// = factored readout inj = B phi(A shat) with A [d x H], B [H x d]
    /// (ENRICHER.md, variants v2.1).
    #[arg(long, default_value_t = 0)]
    pub train_enrich_rank: usize,

    /// Nonlinearity inside the bottleneck of a factored readout
    /// (rank > 0): none = V1 (linear), gelu = V2 (GeluPytorchTanh, matching
    /// the backbone MLP activation).
    #[arg(long, value_enum, default_value = "none")]
    pub train_enrich_act: EnrichAct,

    /// Per-bottleneck-coordinate content gate conditioned on the incoming
    /// token (V3; requires rank > 0): u = sigmoid(q Wu + U z + b), inj =
    /// g B(u * z); the injection penalty stays on the pre-u path.
    #[arg(long, default_value_t = false)]
    pub train_enrich_content_gate: bool,

    /// Treatment of the read-side matrix A (rank > 0): trainable = trained
    /// jointly with B; fixed = fixed random rows ~ N(0, 1/H); pca = set once
    /// from a PCA of collected states via a no-grad calibration pass.
    #[arg(long, value_enum, default_value = "trainable")]
    pub train_enrich_a: EnrichAMode,

    /// Readout-source treatment (ENRICHER.md, Claim R): pick = the readout
    /// sees the pick-orthogonalized state `rmsnorm(shat − alpha·q)` with
    /// alpha = <shat, q>/||q||² (the thought-residual — the state minus its
    /// projection onto the fed token's direction); none = the raw state
    /// (v2.1). Parameter-free; the gate keeps the raw-state view.
    #[arg(long, value_enum, default_value = "none")]
    pub train_enrich_residual: EnrichResidual,
}

impl TrainArgs {
    fn enrich_residual(&self) -> enrich::ResidualMode {
        match self.train_enrich_residual {
            EnrichResidual::None => enrich::ResidualMode::None,
            EnrichResidual::Pick => enrich::ResidualMode::Pick,
        }
    }

    fn enricher_spec(&self) -> enrich::EnricherSpec {
        enrich::EnricherSpec {
            rank: self.train_enrich_rank,
            gelu: matches!(self.train_enrich_act, EnrichAct::Gelu),
            content_gate: self.train_enrich_content_gate,
            a_mode: match self.train_enrich_a {
                EnrichAMode::Trainable => enrich::AMode::Trainable,
                EnrichAMode::Fixed => enrich::AMode::FixedRandom,
                EnrichAMode::Pca => enrich::AMode::Pca,
            },
            residual: self.enrich_residual(),
        }
    }
}

impl From<&TrainArgs> for train::TrainConfig {
    fn from(t: &TrainArgs) -> Self {
        Self {
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
            spec: t.enricher_spec(),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum EnrichAct {
    None,
    Gelu,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum EnrichAMode {
    Trainable,
    Fixed,
    Pca,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum EnrichResidual {
    None,
    Pick,
}

impl Args {
    pub fn feedback_config(&self) -> FeedbackConfig {
        let fb = &self.feedback_args;
        FeedbackConfig {
            mode: fb.softmax_feedback_mode,
            temperature: fb.softmax_feedback_temperature,
            target_entropy: fb.softmax_feedback_target_entropy,
            band: fb.softmax_feedback_band,
            decay: fb.softmax_feedback_decay,
            max_cosine: fb.softmax_feedback_max_cosine,
            blend: fb.softmax_feedback_blend,
            prefill_blend: fb.prefill_feedback_blend,
            prefill_iters: fb.prefill_feedback_iters,
        }
    }

    pub fn enrich_residual(&self) -> enrich::ResidualMode {
        self.train_args.enrich_residual()
    }

    pub fn enricher_spec(&self) -> enrich::EnricherSpec {
        self.train_args.enricher_spec()
    }
}
