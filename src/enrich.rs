use std::path::Path;

use anyhow::{Context, Result};
use candle::{D, DType, Device, Result as CResult, Tensor};

/// Gated linear state readout (v2): enriches a token embedding with a gated,
/// learned linear function of the *previous processing step's output state*
/// (single-state recurrence, no buffer). The backbone is frozen; only this
/// module is trained.
///
/// All weights zero- or neutral-initialized so that a freshly constructed
/// enricher is the identity: the injection `g * W shat` starts at exactly
/// zero (W = 0), i.e. the model starts bit-exact at baseline.
///
/// The recurrence is trained as an RNN with truncated BPTT (see ENRICHER.md).
pub struct Enricher {
    varmap: candle_nn::VarMap,
    pub w: Tensor, // [hidden, hidden], zero-init
    pub gate_w: Tensor, // [2 * hidden]
    pub gate_b: Tensor, // [1]
    hidden: usize,
    /// The previous processing step's pre-final-norm output state, [hidden].
    prev_state: Option<Tensor>,
    /// Gate-value accumulator for monitoring (interior mutability because
    /// enrich takes &self).
    gate_sum: std::cell::Cell<f64>,
    gate_n: std::cell::Cell<usize>,
    /// Training aid: floor for the gate value (set > 0 during warmup so a
    /// mis-step cannot switch the module off and zero the W gradient).
    gate_floor: std::cell::Cell<f64>,
}

const NAMES: [&str; 3] = ["enrich.w", "enrich.gate_w", "enrich.gate_b"];

impl Enricher {
    pub fn new(hidden: usize, gate_bias: f64, _device: &Device) -> CResult<Self> {
        use candle_nn::init::Init;
        let varmap = candle_nn::VarMap::new();
        let w = varmap.get(
            (hidden, hidden),
            NAMES[0],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_w = varmap.get(
            2 * hidden,
            NAMES[1],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_b = varmap.get(
            1,
            NAMES[2],
            Init::Const(gate_bias),
            DType::F32,
            &Device::Cpu,
        )?;
        Ok(Self {
            varmap,
            w,
            gate_w,
            gate_b,
            hidden,
            prev_state: None,
            gate_sum: std::cell::Cell::new(0.0),
            gate_n: std::cell::Cell::new(0),
            gate_floor: std::cell::Cell::new(0.0),
        })
    }

    /// Load weights from a safetensors checkpoint produced by `save`. The
    /// parameters are registered in a fresh VarMap first (so they are
    /// Var-backed and trainable) and then overwritten from the checkpoint.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tensors = candle::safetensors::load(path, &Device::Cpu)?;
        let get_shape = |name: &str| -> Result<usize> {
            let t = tensors
                .get(name)
                .with_context(|| format!("missing {name} in {}", path.display()))?;
            Ok(t.dim(0)?)
        };
        let hidden = get_shape(NAMES[0])?;
        let gate_w_len = get_shape(NAMES[1])?;
        use candle_nn::init::Init;
        let mut varmap = candle_nn::VarMap::new();
        let w = varmap.get(
            (hidden, hidden),
            NAMES[0],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_w = varmap.get(
            gate_w_len,
            NAMES[1],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_b = varmap.get(
            1,
            NAMES[2],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        varmap.load(path)?;
        Ok(Self {
            varmap,
            w,
            gate_w,
            gate_b,
            hidden,
            prev_state: None,
            gate_sum: std::cell::Cell::new(0.0),
            gate_n: std::cell::Cell::new(0),
            gate_floor: std::cell::Cell::new(0.0),
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let map: std::collections::HashMap<String, Tensor> = NAMES
            .iter()
            .map(|n| n.to_string())
            .zip([self.w.clone(), self.gate_w.clone(), self.gate_b.clone()])
            .collect();
        candle::safetensors::save(&map, path.as_ref())?;
        Ok(())
    }

    /// Trainable variables (for the optimizer).
    pub fn vars(&self) -> Vec<candle::Var> {
        self.varmap.all_vars()
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

    /// The learnable enrichment, batched: `x = q + g * W shat(s)` where
    /// `q = e * sqrt(h)` and `shat = s / rms(s)`. `e` and `s` are [B, hidden]
    /// (raw, unscaled embedding rows and raw output states; `s = None` on the
    /// first step). Returns the enriched input [B, hidden], the pre-gate
    /// relative injection energy `||W shat||^2 / ||q||^2` [B, 1] (the
    /// training-time penalty target; cannot be dodged by closing the gate)
    /// and the applied (post-gate) relative injection energy [B, 1]
    /// (monitoring only).
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
        let inj = sh.matmul(&self.w)?; // [B, H]
        let gate_in = Tensor::cat(&[&q, &sh], D::Minus1)?; // [B, 2H]
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
        self.gate_sum.set(self.gate_sum.get() + gate.sum_all()?.to_scalar::<f32>()? as f64);
        self.gate_n.set(self.gate_n.get() + gate.dim(0)?);
        let applied = inj.broadcast_mul(&gate)?; // [B, H]
        let preg = inj.sqr()?.sum_keepdim(D::Minus1)?.broadcast_div(&q_sq)?; // [B, 1]
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

    /// Set the gate floor used in enrich() (training warmup aid).
    pub fn set_gate_floor(&self, floor: f64) {
        self.gate_floor.set(floor);
    }

    /// Mean gate value since the last call (monitoring).
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
}
