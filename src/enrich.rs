use std::path::Path;

use anyhow::Result;
use candle::{D, DType, Device, Result as CResult, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// How the read-side matrix A of a factored (rank-bottleneck) readout is
/// treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AMode {
    /// A is trained jointly with B (default for factored variants).
    Trainable,
    /// A is a fixed random projection (rows ~ N(0, 1/H), ~unit rows): the
    /// ELM/reservoir reading; only B is trained.
    FixedRandom,
    /// A is set once from a PCA of collected RMS-normalized states, then
    /// fixed; only B is trained. Requires a calibration pass on a fresh run.
    Pca,
}

/// The readout-source treatment (ENRICHER.md, Claim R "thought-residual").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualMode {
    /// The v2.1 readout: A reads the raw RMS-normalized state.
    None,
    /// Pick-orthogonal source: `shat_r = rmsnorm(shat − alpha·q_t)` with
    /// `alpha = <shat, q_t>/||q_t||²` (the projection coefficient — on-manifold
    /// ~cos(state, pick)). The readout reads what did not survive sampling;
    /// the gate keeps its raw-state view. Parameter-free.
    Pick,
}

/// Enricher variant specification (ENRICHER.md, "Enricher variants v2.1").
///
/// - `rank == 0`: the v2 full-rank gated readout, `inj = W shat`, W in
///   R^{H x H}, zero-init (bit-exact baseline at init). The default.
/// - `rank == d > 0`: the factored readout `inj = B phi(A shat)` with
///   A in R^{d x H}, B in R^{H x d}; B is zero-init (same identity-at-init
///   and B-first wakeup order), A rows are ~unit vectors. `phi` is GELU when
///   `gelu` is set (variant V2), identity otherwise (V1).
/// - `content_gate` (variant V3, rank > 0 only): per-bottleneck-coordinate
///   gate conditioned on the incoming token,
///   `u = sigmoid(q W_u + u_z * z + b_u)` in (0,1)^d with W_u in R^{d x H},
///   u_z in R^d, and `inj = g * B(u * z)`. The injection penalty stays on the
///   PRE-u path `B z`, so the content gate cannot dodge it (M1 lesson a).
/// - `residual`: the readout-source treatment (Claim R): `Pick` feeds the
///   pick-orthogonalized state to the readout (and to the PCA collector);
///   the gate still sees the raw state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnricherSpec {
    pub rank: usize,
    pub gelu: bool,
    pub content_gate: bool,
    pub a_mode: AMode,
    pub residual: ResidualMode,
}

impl Default for EnricherSpec {
    fn default() -> Self {
        Self {
            rank: 0,
            gelu: false,
            content_gate: false,
            a_mode: AMode::Trainable,
            residual: ResidualMode::None,
        }
    }
}

impl EnricherSpec {
    pub fn validate(&self, hidden: usize) -> anyhow::Result<()> {
        if self.rank == 0 {
            anyhow::ensure!(
                !self.gelu && !self.content_gate && self.a_mode == AMode::Trainable,
                "rank 0 (v0 full-rank) cannot be combined with act/content-gate/a-mode options"
            );
            return Ok(());
        }
        anyhow::ensure!(self.rank < hidden, "rank {} must be < hidden {hidden}", self.rank);
        Ok(())
    }

    pub fn summary(&self) -> String {
        if self.rank == 0 {
            return format!("v0 full-rank W ({}x{})", 640, 640);
        }
        format!(
            "factored rank={} phi={} content-gate={} a={:?} residual={:?}",
            self.rank,
            if self.gelu { "gelu" } else { "id" },
            self.content_gate,
            self.a_mode,
            self.residual
        )
    }
}

const CFG_NAME: &str = "enrich.cfg";
const U_BIAS_INIT: f64 = 2.0;

/// Enricher: gated state readout. See the type docs on [`EnricherSpec`] for
/// the variants; ENRICHER.md for the training procedure (RNN TBPTT cascade).
///
/// All weights are zero- or neutral-initialized so that a freshly constructed
/// enricher is the identity: the injection starts at exactly zero (W = 0 resp.
/// B = 0), i.e. the model starts bit-exact at baseline. The wakeup order is
/// B (or W) -> (A, u) -> gate: at B = 0 only dL/dB != 0; the read side and the
/// gate start learning once the injection exists.
pub struct Enricher {
    varmap: candle_nn::VarMap,
    spec: EnricherSpec,
    hidden: usize,
    /// V0 only: [hidden, hidden], zero-init.
    w: Option<Tensor>,
    /// Factored only: A [rank, hidden] (~unit rows), B [hidden, rank],
    /// zero-init.
    a: Option<Tensor>,
    b: Option<Tensor>,
    /// Content gate (V3 only): W_u [rank, hidden], u_z [rank], u_b [rank].
    u_w: Option<Tensor>,
    u_z: Option<Tensor>,
    u_b: Option<Tensor>,
    /// Scalar gate: [2 * hidden] / [1] in every variant.
    gate_w: Tensor,
    gate_b: Tensor,
    /// The previous processing step's pre-final-norm output state, [hidden].
    prev_state: Option<Tensor>,
    /// Gate-value accumulator for monitoring (interior mutability because
    /// enrich takes &self).
    gate_sum: std::cell::Cell<f64>,
    gate_n: std::cell::Cell<usize>,
    /// Content-gate (u) mean accumulator for monitoring (V3 only).
    u_sum: std::cell::Cell<f64>,
    u_n: std::cell::Cell<usize>,
    /// Residual-mode monitors (Claim R): mean |alpha| and mean alpha^2
    /// (the removed energy fraction), plus near-degenerate rows
    /// (alpha^2 > 0.9999). All zero unless `residual == Pick`.
    alpha_abs_sum: std::cell::Cell<f64>,
    alpha_sq_sum: std::cell::Cell<f64>,
    alpha_n: std::cell::Cell<usize>,
    alpha_degen: std::cell::Cell<usize>,
    /// Training aid: floor for the scalar gate value (set > 0 during warmup so
    /// a mis-step cannot switch the module off and zero the B/W gradient).
    gate_floor: std::cell::Cell<f64>,
    /// Detached-shat Gram collector for the PCA calibration of A
    /// (`a_mode == Pca`): accumulates sum(sh sh^T) over enriched positions
    /// while `collect_shat` is on.
    collect_shat: std::cell::Cell<bool>,
    gram: std::cell::RefCell<Option<Tensor>>,
    gram_n: std::cell::Cell<usize>,
}

/// Deterministic Box-Muller normal fill ([rows, cols] f32, seeded) — used for
/// the A-matrix init so that fresh enrichers are reproducible per seed
/// (candle's own Init::Randn uses an unseeded internal RNG).
pub(crate) fn randn_rows(rows: usize, cols: usize, std: f64, seed: u64) -> CResult<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = rows * cols;
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let u1 = (rng.random::<f64>() + 1e-12).min(1.0);
        let u2 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        v.push((r * th.cos() * std) as f32);
        if v.len() < n {
            v.push((r * th.sin() * std) as f32);
        }
    }
    Tensor::from_vec(v, (rows, cols), &Device::Cpu)
}

fn cfg_tensor(spec: &EnricherSpec) -> CResult<Tensor> {
    let a_mode = match spec.a_mode {
        AMode::Trainable => 0.0f32,
        AMode::FixedRandom => 1.0,
        AMode::Pca => 2.0,
    };
    let residual = match spec.residual {
        ResidualMode::None => 0.0f32,
        ResidualMode::Pick => 1.0,
    };
    Tensor::from_vec(
        vec![
            spec.rank as f32,
            if spec.gelu { 1.0 } else { 0.0 },
            if spec.content_gate { 1.0 } else { 0.0 },
            a_mode,
            residual,
        ],
        (5,),
        &Device::Cpu,
    )
}

fn spec_from_cfg(t: &Tensor) -> CResult<EnricherSpec> {
    let v = t.to_vec1::<f32>()?;
    // len 4: the v2.1 cfg (residual did not exist => None); len 5: v2.2+.
    if v.len() != 4 && v.len() != 5 {
        candle::bail!("malformed {} tensor (len {})", CFG_NAME, v.len());
    }
    let a_mode = match v[3] as usize {
        1 => AMode::FixedRandom,
        2 => AMode::Pca,
        _ => AMode::Trainable,
    };
    let residual = if v.len() == 5 && v[4] > 0.5 {
        ResidualMode::Pick
    } else {
        ResidualMode::None
    };
    Ok(EnricherSpec {
        rank: v[0] as usize,
        gelu: v[1] > 0.5,
        content_gate: v[2] > 0.5,
        a_mode,
        residual,
    })
}

impl Enricher {
    pub fn new(
        hidden: usize,
        gate_bias: f64,
        spec: EnricherSpec,
        seed: u64,
    ) -> CResult<Self> {
        spec.validate(hidden).map_err(candle::Error::msg)?;
        let mut varmap = candle_nn::VarMap::new();
        let gate_w = varmap.get(
            2 * hidden,
            "enrich.gate_w",
            candle_nn::init::Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_b = varmap.get(
            1,
            "enrich.gate_b",
            candle_nn::init::Init::Const(gate_bias),
            DType::F32,
            &Device::Cpu,
        )?;
        let (w, a, b, u_w, u_z, u_b) = if spec.rank == 0 {
            let w = varmap.get(
                (hidden, hidden),
                "enrich.w",
                candle_nn::init::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )?;
            (Some(w), None, None, None, None, None)
        } else {
            let d = spec.rank;
            let a = varmap.get(
                (d, hidden),
                "enrich.a",
                candle_nn::init::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )?;
            if spec.a_mode != AMode::Pca {
                // Deterministic per-seed rows ~ N(0, 1/H): row norms ~ 1, so
                // z = A shat has ~unit-std coordinates (GELU-healthy regime).
                let fill = randn_rows(d, hidden, (hidden as f64).sqrt().recip(), seed)?;
                varmap.set_one("enrich.a", &fill)?;
            }
            // Pca: left zero until the calibration pass fills it.
            let b = varmap.get(
                (hidden, d),
                "enrich.b",
                candle_nn::init::Init::Const(0.),
                DType::F32,
                &Device::Cpu,
            )?;
            let (u_w, u_z, u_b) = if spec.content_gate {
                let u_w = varmap.get(
                    (d, hidden),
                    "enrich.u_w",
                    candle_nn::init::Init::Const(0.),
                    DType::F32,
                    &Device::Cpu,
                )?;
                let u_z = varmap.get(
                    d,
                    "enrich.u_z",
                    candle_nn::init::Init::Const(0.),
                    DType::F32,
                    &Device::Cpu,
                )?;
                let u_b = varmap.get(
                    d,
                    "enrich.u_b",
                    candle_nn::init::Init::Const(U_BIAS_INIT),
                    DType::F32,
                    &Device::Cpu,
                )?;
                (Some(u_w), Some(u_z), Some(u_b))
            } else {
                (None, None, None)
            };
            (None, Some(a), Some(b), u_w, u_z, u_b)
        };
        Ok(Self {
            varmap,
            spec,
            hidden,
            w,
            a,
            b,
            u_w,
            u_z,
            u_b,
            gate_w,
            gate_b,
            prev_state: None,
            gate_sum: std::cell::Cell::new(0.0),
            gate_n: std::cell::Cell::new(0),
            u_sum: std::cell::Cell::new(0.0),
            u_n: std::cell::Cell::new(0),
            alpha_abs_sum: std::cell::Cell::new(0.0),
            alpha_sq_sum: std::cell::Cell::new(0.0),
            alpha_n: std::cell::Cell::new(0),
            alpha_degen: std::cell::Cell::new(0),
            gate_floor: std::cell::Cell::new(0.0),
            collect_shat: std::cell::Cell::new(false),
            gram: std::cell::RefCell::new(None),
            gram_n: std::cell::Cell::new(0),
        })
    }

    /// Load weights from a safetensors checkpoint produced by `save`. The
    /// variant is read from the self-describing `enrich.cfg` tensor; a
    /// checkpoint without it is the v0 full-rank format. Parameters are
    /// registered in a fresh VarMap first (so they are Var-backed and
    /// trainable) and then overwritten from the checkpoint.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tensors = candle::safetensors::load(path, &Device::Cpu)?;
        let spec = match tensors.get(CFG_NAME) {
            Some(t) => spec_from_cfg(t)?,
            None => EnricherSpec::default(),
        };
        let hidden = if let Some(t) = tensors.get("enrich.w") {
            t.dim(0)?
        } else if let Some(t) = tensors.get("enrich.a") {
            t.dim(1)?
        } else {
            anyhow::bail!("no enrich.w / enrich.a in {}", path.display());
        };
        let mut enr = Self::new(hidden, 0.0, spec, 0)?;
        enr.varmap.load(path)?;
        Ok(enr)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert(CFG_NAME.to_string(), cfg_tensor(&self.spec)?);
        if let Some(w) = &self.w {
            map.insert("enrich.w".to_string(), w.clone());
        }
        if let Some(a) = &self.a {
            map.insert("enrich.a".to_string(), a.clone());
        }
        if let Some(b) = &self.b {
            map.insert("enrich.b".to_string(), b.clone());
        }
        if let (Some(uw), Some(uz), Some(ub)) = (&self.u_w, &self.u_z, &self.u_b) {
            map.insert("enrich.u_w".to_string(), uw.clone());
            map.insert("enrich.u_z".to_string(), uz.clone());
            map.insert("enrich.u_b".to_string(), ub.clone());
        }
        map.insert("enrich.gate_w".to_string(), self.gate_w.clone());
        map.insert("enrich.gate_b".to_string(), self.gate_b.clone());
        candle::safetensors::save(&map, path.as_ref())?;
        Ok(())
    }

    pub fn spec(&self) -> &EnricherSpec {
        &self.spec
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Ordered (name, tensor) list of every trainable parameter tensor of the
    /// active variant (for gradient logging / sweeps).
    pub fn named_params(&self) -> Vec<(&'static str, &Tensor)> {
        let mut out: Vec<(&'static str, &Tensor)> = Vec::new();
        if self.spec.rank == 0 {
            out.push(("w", self.w.as_ref().unwrap()));
        } else {
            out.push(("b", self.b.as_ref().unwrap()));
            if self.spec.a_mode == AMode::Trainable {
                out.push(("a", self.a.as_ref().unwrap()));
            }
            if self.spec.content_gate {
                out.push(("u_w", self.u_w.as_ref().unwrap()));
                out.push(("u_z", self.u_z.as_ref().unwrap()));
                out.push(("u_b", self.u_b.as_ref().unwrap()));
            }
        }
        out.push(("gate_w", &self.gate_w));
        out.push(("gate_b", &self.gate_b));
        out
    }

    /// The trainable variables (for the optimizer / clipping): every
    /// parameter of the active variant except a fixed or PCA-set A.
    pub fn trainable_vars(&self) -> Vec<candle::Var> {
        let data = self.varmap.data().lock().unwrap();
        let mut names: Vec<String> = self
            .named_params()
            .iter()
            .map(|(n, _)| format!("enrich.{n}"))
            .collect();
        names.sort(); // deterministic order
        names
            .into_iter()
            .map(|n| data.get(&n).unwrap_or_else(|| panic!("{n} missing from varmap")).clone())
            .collect()
    }

    pub fn clear_state(&mut self) {
        self.prev_state = None;
    }

    /// Store the current processing step's output state ([hidden]).
    pub fn push_state(&mut self, state: Tensor) {
        self.prev_state = Some(state);
    }

    pub fn prev_state(&self) -> Option<&Tensor> {
        self.prev_state.as_ref()
    }

    /// Enable/disable the detached-shat Gram collector (PCA calibration).
    /// Enabling resets any partial accumulation; disabling keeps it for
    /// `take_gram()`.
    pub fn set_collect_shat(&self, on: bool) {
        self.collect_shat.set(on);
        if on {
            *self.gram.borrow_mut() = None;
            self.gram_n.set(0);
        }
    }

    /// Take the accumulated Gram matrix sum(sh sh^T) [H, H] and the number of
    /// collected rows (resets the collector).
    pub fn take_gram(&self) -> Option<(Tensor, usize)> {
        let g = self.gram.borrow_mut().take()?;
        let n = self.gram_n.get();
        self.gram_n.set(0);
        Some((g, n))
    }

    /// Overwrite the A matrix (PCA calibration; rank > 0 only).
    pub fn set_a(&mut self, a: &Tensor) -> CResult<()> {
        if self.spec.rank == 0 {
            candle::bail!("set_a on a full-rank (v0) enricher");
        }
        self.varmap.set_one("enrich.a", a)?;
        Ok(())
    }

    /// The injection branch, pre-scalar-gate: returns (inj_pre, applied_pre).
    /// `inj_pre` is the penalty path (never gated by u); `applied_pre` is what
    /// the scalar gate multiplies (u-masked in V3, identical otherwise).
    fn compute_injection(&self, q: &Tensor, sh: &Tensor) -> CResult<(Tensor, Tensor)> {
        if self.spec.rank == 0 {
            let inj = sh.matmul(self.w.as_ref().unwrap())?; // [B, H]
            return Ok((inj.clone(), inj));
        }
        let d = self.spec.rank;
        let a_read = match self.spec.a_mode {
            AMode::Trainable => self.a.as_ref().unwrap().clone(),
            _ => self.a.as_ref().unwrap().detach(),
        };
        let mut z = sh.matmul(&a_read.t()?)?; // [B, d]
        if self.spec.gelu {
            z = z.gelu()?;
        }
        if let Some(u_w) = &self.u_w {
            let u_z = self.u_z.as_ref().unwrap();
            let u_b = self.u_b.as_ref().unwrap();
            let qterm = q.matmul(&u_w.t()?)?; // [B, d]
            let zterm = z.broadcast_mul(&u_z.reshape((1, d))?)?; // [B, d]
            let u = candle_nn::ops::sigmoid(
                &qterm
                    .broadcast_add(&zterm)?
                    .broadcast_add(&u_b.reshape((1, d))?)?,
            )?; // [B, d]
            self.u_sum
                .set(self.u_sum.get() + u.sum_all()?.to_scalar::<f32>()? as f64);
            // u has `rank` gates per row; normalize the monitor mean by the
            // element count, not the row count.
            self.u_n.set(self.u_n.get() + u.elem_count());
            let b_t = self.b.as_ref().unwrap().t()?;
            let inj_pre = z.matmul(&b_t)?; // [B, H] — the tax path (pre-u)
            let applied_pre = z.broadcast_mul(&u)?.matmul(&b_t)?; // [B, H]
            Ok((inj_pre, applied_pre))
        } else {
            let inj = z.matmul(&self.b.as_ref().unwrap().t()?)?; // [B, H]
            Ok((inj.clone(), inj))
        }
    }

    /// The learnable enrichment, batched: `x = q + g * inj` where
    /// `q = e * sqrt(h)`, `shat = s / rms(s)`, and `inj` is
    /// `g * phi_u(W shat)` (v0) or `g * u(?) * B phi(A shat)` (factored
    /// variants). With `residual == Pick` the readout (and PCA collector)
    /// sees `shat_r = rmsnorm(shat − alpha·q)` instead — the pick-orthogonal
    /// thought-residual (Claim R); the gate keeps the raw-state view.
    /// `e` and `s` are [B, hidden] (raw, unscaled embedding rows
    /// and raw output states; `s = None` on the first step). Returns the
    /// enriched input [B, hidden], the pre-gate relative injection energy
    /// `||inj_pre||^2 / ||q||^2` [B, 1] (the training-time penalty target;
    /// cannot be dodged by closing the scalar gate or the content gate) and
    /// the applied (post-gate) relative injection energy [B, 1] (monitoring
    /// only).
    pub fn enrich(
        &self,
        e: &Tensor,
        prev_state: Option<&Tensor>,
    ) -> CResult<(Tensor, Tensor, Tensor)> {
        let scale = (self.hidden as f64).sqrt();
        let q = e.to_dtype(DType::F32)?.affine(scale, 0.0)?; // [B, H]
        let q_sq = q.sqr()?.sum_keepdim(D::Minus1)?; // [B, 1]
        let zero = q_sq.affine(0.0, 0.0)?; // [B, 1] zeros
        let Some(s) = prev_state else {
            return Ok((q, zero.clone(), zero));
        };
        let sh = s.to_dtype(DType::F32)?.broadcast_div(
            &((s.to_dtype(DType::F32)?
                .sqr()?
                .sum_keepdim(D::Minus1)?
                / self.hidden as f64)?
                + 1e-6)?
                .sqrt()?,
        )?; // [B, H], ||shat|| = sqrt(H)
        // Claim R (thought-residual): pick-orthogonal readout source.
        // alpha = <sh, q>/||q||^2 per row — the projection coefficient of the
        // state onto the fed token's direction (~cos(state, pick) on-manifold,
        // i.e. the model's own confidence in the pick, by the head's geometry).
        // h = sh − alpha·q is the orthogonal complement of the pick direction;
        // re-RMS-normalized with the same +1e-6 epsilon, so a degenerate
        // residual (alpha^2 -> 1) becomes a ~zero readout source instead of a
        // division blowup. The readout and the PCA collector see sh_read; the
        // gate below keeps the raw-state view.
        let sh_read = match self.spec.residual {
            ResidualMode::Pick => {
                let alpha = sh
                    .broadcast_mul(&q)?
                    .sum_keepdim(D::Minus1)? // [B, 1]
                    .broadcast_div(&q_sq)?; // [B, 1]
                let av = alpha.flatten_all()?.to_vec1::<f32>()?;
                let mut abs_sum = 0f64;
                let mut sq_sum = 0f64;
                let mut degen = 0usize;
                for &a in &av {
                    let a = a as f64;
                    abs_sum += a.abs();
                    sq_sum += a * a;
                    if a * a > 0.9999 {
                        degen += 1;
                    }
                }
                self.alpha_abs_sum.set(self.alpha_abs_sum.get() + abs_sum);
                self.alpha_sq_sum.set(self.alpha_sq_sum.get() + sq_sum);
                self.alpha_n.set(self.alpha_n.get() + av.len());
                self.alpha_degen.set(self.alpha_degen.get() + degen);
                let h = sh.broadcast_sub(&alpha.broadcast_mul(&q)?)?; // [B, H]
                h.broadcast_div(
                    &((h.sqr()?.sum_keepdim(D::Minus1)? / self.hidden as f64)? + 1e-6)?.sqrt()?,
                )? // [B, H], ||shat_r|| = sqrt(H)
            }
            ResidualMode::None => sh.clone(),
        };
        if self.collect_shat.get() {
            let shd = sh_read.detach();
            let g = shd.t()?.matmul(&shd)?; // [H, H]
            let mut slot = self.gram.borrow_mut();
            match slot.as_ref() {
                Some(acc) => {
                    let acc = acc.add(&g)?;
                    *slot = Some(acc);
                }
                None => *slot = Some(g),
            }
            self.gram_n.set(self.gram_n.get() + sh.dim(0)?);
        }
        let (inj_pre, applied_pre) = self.compute_injection(&q, &sh_read)?;
        let gate_in = Tensor::cat(&[&q, &sh], D::Minus1)?; // [B, 2H] raw-state view
        let gate = candle_nn::ops::sigmoid(
            &gate_in
                .matmul(&self.gate_w.unsqueeze(1)?)?
                .broadcast_add(&self.gate_b.reshape((1, 1))?)?,
        )?; // [B, 1]
        let floor = self.gate_floor.get();
        let gate = if floor > 0.0 {
            gate.affine(1.0 - floor, floor)?
        } else {
            gate
        };
        self.gate_sum
            .set(self.gate_sum.get() + gate.sum_all()?.to_scalar::<f32>()? as f64);
        self.gate_n.set(self.gate_n.get() + gate.dim(0)?);
        let applied = applied_pre.broadcast_mul(&gate)?; // [B, H]
        let preg = inj_pre.sqr()?.sum_keepdim(D::Minus1)?.broadcast_div(&q_sq)?; // [B, 1]
        let appl = applied.sqr()?.sum_keepdim(D::Minus1)?.broadcast_div(&q_sq)?; // [B, 1]
        let x = q.broadcast_add(&applied)?; // [B, H]
        Ok((x, preg, appl))
    }

    /// Single-row convenience wrapper ([hidden] in, [hidden] x out, pre-gate
    /// relative injection energy as a scalar).
    pub fn enrich_1d(
        &self,
        e: &Tensor,
        prev_state: Option<&Tensor>,
    ) -> CResult<(Tensor, f64)> {
        let e2 = e.reshape((1, e.dims1()?))?;
        let s2 = match prev_state {
            Some(s) => Some(s.reshape((1, s.dims1()?))?),
            None => None,
        };
        let (x, preg, _appl) = self.enrich(&e2, s2.as_ref())?;
        let preg = preg.squeeze(0)?.squeeze(0)?.to_scalar::<f32>()? as f64;
        Ok((x.squeeze(0)?, preg))
    }

    /// Set the scalar-gate floor used in enrich() (training warmup aid).
    pub fn set_gate_floor(&self, floor: f64) {
        self.gate_floor.set(floor);
    }

    /// Mean scalar-gate value since the last call (monitoring).
    pub fn take_gate_mean(&self) -> Option<f64> {
        let n = self.gate_n.get();
        if n == 0 {
            return None;
        }
        let m = self.gate_sum.get() / n as f64;
        self.gate_sum.set(0.0);
        self.gate_n.set(0);
        Some(m)
    }

    /// Mean content-gate (u) value since the last call (monitoring; V3 only).
    pub fn take_content_gate_mean(&self) -> Option<f64> {
        let n = self.u_n.get();
        if n == 0 {
            return None;
        }
        let m = self.u_sum.get() / n as f64;
        self.u_sum.set(0.0);
        self.u_n.set(0);
        Some(m)
    }

    /// Residual-mode monitors since the last call (Claim R): (mean |alpha|,
    /// mean alpha^2 = removed energy fraction, near-degenerate rows). None
    /// unless `residual == Pick` and at least one enriched step ran.
    pub fn take_residual_stats(&self) -> Option<(f64, f64, usize)> {
        let n = self.alpha_n.get();
        if n == 0 {
            return None;
        }
        let abs = self.alpha_abs_sum.get() / n as f64;
        let rem = self.alpha_sq_sum.get() / n as f64;
        let degen = self.alpha_degen.get();
        self.alpha_abs_sum.set(0.0);
        self.alpha_sq_sum.set(0.0);
        self.alpha_n.set(0);
        self.alpha_degen.set(0);
        Some((abs, rem, degen))
    }
}
