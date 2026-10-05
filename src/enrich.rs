use std::path::Path;

use anyhow::{Context, Result};
use candle::{DType, Device, D, Result as CResult, Tensor};

/// Attention-pooling "thought pool" enricher: enriches a token embedding
/// with a learned, gated mixture of the output states of already processed
/// tokens (previous decode steps, or causally previous positions during
/// prefill). The backbone is frozen; only this module is trained.
///
/// All weights zero- or neutral-initialized so that a freshly constructed
/// enricher is the identity: the injection `g * W_o(pooled)` starts at
/// exactly zero (W_o = 0), i.e. the model starts bit-exact at baseline.
pub struct Enricher {
    varmap: candle_nn::VarMap,
    pub w_q: Tensor, // [hidden, r]
    pub w_k: Tensor, // [hidden, r]
    pub w_v: Tensor, // [hidden, r]
    pub w_o: Tensor, // [r, hidden]
    pub gate_w: Tensor, // [2 * hidden]
    pub gate_b: Tensor, // [1]
    hidden: usize,
    r: usize,
    /// Ring buffer of the last `capacity` output states, oldest first.
    states: Vec<Tensor>,
    capacity: usize,
    /// Gate-value accumulator for monitoring (interior mutability because
    /// enrich takes &self).
    gate_sum: std::cell::Cell<f64>,
    gate_n: std::cell::Cell<usize>,
    /// Training aid: floor for the gate value (set > 0 during warmup so a
    /// mis-step cannot switch the module off and zero the W_o gradient).
    gate_floor: std::cell::Cell<f64>,
}

const NAMES: [&str; 6] = [
    "enrich.w_q",
    "enrich.w_k",
    "enrich.w_v",
    "enrich.w_o",
    "enrich.gate_w",
    "enrich.gate_b",
];

impl Enricher {
    pub fn new(hidden: usize, r: usize, capacity: usize, device: &Device) -> CResult<Self> {
        use candle_nn::init::Init;
        let varmap = candle_nn::VarMap::new();
        let scale = (hidden as f64).sqrt();
        let w_q = varmap.get(
            (hidden, r),
            NAMES[0],
            Init::Uniform { lo: -1.0 / scale, up: 1.0 / scale },
            DType::F32,
            &Device::Cpu,
        )?;
        let w_k = varmap.get(
            (hidden, r),
            NAMES[1],
            Init::Uniform { lo: -1.0 / scale, up: 1.0 / scale },
            DType::F32,
            &Device::Cpu,
        )?;
        let w_v = varmap.get(
            (hidden, r),
            NAMES[2],
            Init::Uniform { lo: -1.0 / scale, up: 1.0 / scale },
            DType::F32,
            &Device::Cpu,
        )?;
        let w_o = varmap.get(
            (r, hidden),
            NAMES[3],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_w = varmap.get(
            2 * hidden,
            NAMES[4],
            Init::Const(0.),
            DType::F32,
            &Device::Cpu,
        )?;
        let gate_b_var = varmap.get(
            1,
            NAMES[5],
            Init::Const(2.),
            DType::F32,
            &Device::Cpu,
        )?;
        let _ = device;
        Ok(Self {
            varmap,
            w_q,
            w_k,
            w_v,
            w_o,
            gate_w,
            gate_b: gate_b_var.clone(),
            hidden,
            r,
            states: Vec::with_capacity(capacity),
            capacity,
            gate_sum: std::cell::Cell::new(0.0),
            gate_n: std::cell::Cell::new(0),
            gate_floor: std::cell::Cell::new(0.0),
        })
    }

    /// Load weights from a safetensors checkpoint produced by `save`.
    pub fn load(path: impl AsRef<Path>, _hidden: usize, _r: usize, capacity: usize) -> Result<Self> {
        let path = path.as_ref();
        let tensors = candle::safetensors::load(path, &Device::Cpu)?;
        let get = |name: &str| -> Result<Tensor> {
            tensors
                .get(name)
                .cloned()
                .with_context(|| format!("missing {name} in {}", path.display()))
        };
        let w_q = get(NAMES[0])?;
        let w_k = get(NAMES[1])?;
        let w_v = get(NAMES[2])?;
        let w_o = get(NAMES[3])?;
        let gate_w = get(NAMES[4])?;
        let gate_b = get(NAMES[5])?;
        let hidden = w_k.dim(0)?;
        let r = w_k.dim(1)?;
        Ok(Self {
            varmap: candle_nn::VarMap::new(),
            w_q,
            w_k,
            w_v,
            w_o,
            gate_w,
            gate_b,
            hidden,
            r,
            states: Vec::with_capacity(capacity),
            capacity,
            gate_sum: std::cell::Cell::new(0.0),
            gate_n: std::cell::Cell::new(0),
            gate_floor: std::cell::Cell::new(0.0),
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let map: std::collections::HashMap<String, Tensor> = NAMES
            .iter()
            .map(|n| n.to_string())
            .zip([
                self.w_q.clone(),
                self.w_k.clone(),
                self.w_v.clone(),
                self.w_o.clone(),
                self.gate_w.clone(),
                self.gate_b.clone(),
            ])
            .collect();
        candle::safetensors::save(&map, path.as_ref())?;
        Ok(())
    }

    /// Trainable variables (for the optimizer).
    pub fn vars(&self) -> Vec<candle::Var> {
        self.varmap.all_vars()
    }

    pub fn clear_states(&mut self) {
        self.states.clear();
    }

    /// Current state window (read-only view into the ring buffer).
    pub fn window(&self) -> &[Tensor] {
        &self.states
    }

    /// Store an output state (the previous processing step's pre-final-norm
    /// hidden vector, shape [hidden]).
    pub fn push_state(&mut self, state: Tensor) {
        if self.states.len() == self.capacity {
            self.states.remove(0);
        }
        self.states.push(state);
    }

    /// The learnable enrichment: `e * sqrt(h) + g * W_o(attend(states))`.
    /// Functional form (explicit state window) used both at inference and
    /// during training; `e` is the raw embedding row [hidden] and the
    /// Gemma embedding scale is applied here.
    pub fn enrich(&self, e: &Tensor, window: &[Tensor]) -> CResult<Tensor> {
        let scale = (self.hidden as f64).sqrt();
        let q = e.to_dtype(DType::F32)?.affine(scale, 0.0)?.reshape((1, self.hidden))?;
        if window.is_empty() {
            return Ok(q.reshape((self.hidden,))?);
        }
        let hs = Tensor::cat(window, D::Minus1)?.reshape((window.len(), self.hidden))?;
        let keys = hs.matmul(&self.w_k)?; // [n, r]
        let vals = hs.matmul(&self.w_v)?; // [n, r]
        let q_proj = q.matmul(&self.w_q)?; // [1, r]
        let scores = (q_proj
            .matmul(&keys.transpose(0, 1)?)?
            .affine(1.0 / (self.r as f64).sqrt(), 0.0))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?; // [1, n]
        let pooled = attn.matmul(&vals)?; // [1, r]
        let injected = pooled.matmul(&self.w_o)?; // [1, hidden]
        let gate_in = Tensor::cat(&[&q, &injected], D::Minus1)?; // [1, 2*hidden]
        let gate = candle_nn::ops::sigmoid(
            &(gate_in.matmul(&self.gate_w.unsqueeze(1)?)? + &self.gate_b.reshape((1, 1))?)?,
        )?; // [1, 1]
        let floor = self.gate_floor.get();
        let gate = if floor > 0.0 {
            gate.affine(1.0 - floor, floor)?
        } else {
            gate
        };
        self.gate_sum.set(
            self.gate_sum.get() + gate.flatten_all()?.to_vec1::<f32>()?[0] as f64,
        );
        self.gate_n.set(self.gate_n.get() + 1);
        let out = injected.broadcast_mul(&gate)?.squeeze(0)?;
        (q.squeeze(0)? + out)?.reshape((self.hidden,))
    }

    /// Mean gate value since the last call (monitoring).
    /// Set the gate floor used in enrich() (training warmup aid).
    pub fn set_gate_floor(&self, floor: f64) {
        self.gate_floor.set(floor);
    }

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
