# Trained state-readout enricher: design, training, results

## Motivation and concept

During one token processing step many different "memories" (associations)
are activated and interact to form the output vector, but that rich mixture
is thrown away at the sampling step — only a single discrete token id
survives, and the next step's first layers (almost) start from scratch.
Later layers retain access to the emergent mixture via the KV cache /
attention, but early layers see only one token embedding. The looped /
universal-transformer line of work suggests that spending more compute on
the same, already-picked concepts (thinking on the same tokens for longer)
improves performance. The aim: **spend more compute on the concepts picked
during the previous token processing, so that ideas that started to emerge
are not lost again and again.**

Findings 6-8 (FEEDBACK.md) sharpened this:

- Expected-embedding feedback damages in-context induction (60% -> 43%):
  the induction head needs the *exact* previous-token identity as its
  attention key; the expected embedding smears that key over the token's
  neighborhood (recall survives, precision dies: "68" -> "6").
- Blending with the sampled token (a=0.5) rescues most of the damage
  (-> 56%) and is uniformly non-harmful — the sampled token re-anchors the
  attention key.
- The iterated hand-set prefill cascade at gentle strength (K=2, a_p=0.25)
  is the only prompt-side mechanism that *improves* ICL (induce 78% vs 60%
  baseline, p=0.006).
- Hand-crafted feedback at fixed strength is uniformly destructive at
  strong settings (Finding 7): strength and content must be **learned, per
  step** — not hand-set.

The resulting mechanism: a minimal **gated state readout** trained as an
RNN with truncated BPTT, deployed as a sequential cascade over both prompt
and answer.

## Architecture (v2.2)

Core readout, at each decode/prompt position t:

```
q_t     = E[a_t] * sqrt(h)                # fed token, on-manifold, ||q|| ~ 25
s_t     = f_theta(x_t)                    # frozen Gemma step (attends KV <= t)
shat_t  = s_t / rms(s_t)                  # RMS-normalized, ||shat|| = sqrt(h)
inj_t   = W  shat_{t-1}                   # W in R^{640x640}, ZERO-INIT
g_t     = sigmoid(w_g [q_t; shat_{t-1}] + b)   # scalar gate, w_g in R^{1280}, b = +2
x_t     = q_t + g_t * inj_t
```

- **410,881 params (~1.6 MB F32)**; grads + AdamW moments ~6.6 MB.
- **Zero-init W** => bit-exact baseline at init (regression property kept
  from M1).
- The state read from is the **single previous step's output state**
  (pre-final-norm last-position hidden) — no ring buffer.
- RMS-normalization makes the readout scale-free (better-conditioned
  gradients; the injection penalty stays meaningful as state norms drift).
- Wakeup: at W=0, dL/dW != 0 immediately but dL/dgate = 0 until W moves —
  the gate starts learning only after the injection exists.
- Training and inference both run the full sequential cascade over prompt
  AND answer; parallel prefill is *never* used when the enricher is
  attached.

### Factored variants (v2.1)

The full-rank W puts 409,600 of its 410,881 params against <= 32 loss
tokens per effective batch — most of W never earns signal (weight decay is
all that happens in its null space). The factored form concentrates the
same learning into 2*d*H entries and doubles as a direct test of the
module's core hypothesis (the thrown-away mixture is a low-dim summary).
Compute is a non-issue either way (~0.3-3.3 MFLOPs/step vs ~540 MFLOPs +
1.08 GB weight streaming for the frozen model), so variant comparisons
cost nothing extra per run.

Variants (selected by `--train-enrich-rank d`, `--train-enrich-act`,
`--train-enrich-content-gate`, `--train-enrich-a`):

```
V0 (default, rank 0):  inj = W shat                        W in R^{640x640}   410,881 params
V1 (rank d):           inj = B A shat                      A [d,H], B [H,d]   d=64: 83,201
V2 (V1 + act):         inj = B gelu(A shat)                same params
V3 (V2 + content gate):u = sigmoid(q Wu + u_z*z + b_u);    + 45,120 (d=64)
                       inj = g * B(u * z)                  128,321 total
```

- The scalar gate is unchanged in all variants; V3 adds *content* selection
  (which bottleneck features to inject for this token) on top of V2's
  *magnitude* selection.
- Init/wakeup identical in every variant: B (or W) zero-init => bit-exact
  baseline; dL/dB != 0 at init, dL/dA = dL/du = dL/dg = 0 until B moves
  (measured fresh-init sweep signatures, batch 2, tau=4: V0 w 16.38,
  V1-d64 b 5.06, V2-d64 b 3.18, V3-d64 b 2.80, PCA-r32 b 16.32 — a/u/gate
  all 0.000 as designed).
- Guards port unchanged: the injection penalty measures the pre-gate,
  pre-u energy `||B phi(A shat)||^2 / ||q||^2` (u cannot dodge it); gelu'
  adds a <= 1.1 factor to the contraction (c^6 headroom intact).
- Read-side A treatment (`--train-enrich-a`): `trainable` (default;
  deterministic per-seed rows ~ N(0, 1/H), ~unit rows), `fixed`
  (ELM/reservoir reading; only B + gates train), `pca` (A set once from
  the top-d eigenvectors of the collected states' Gram matrix, then fixed:
  one no-grad cascade over the first 16 training batches (~2k state rows),
  top-d orthonormalized power iteration (60 rounds) => unit-norm rows
  (z coordinates ~unit variance, GELU-healthy); idempotent on resume,
  ~2-3 min one-time. PCA-A reads state-energy-aligned directions
  (fresh-init dB 16.3 vs 5.1 for random A) — the prior pays off
  immediately in signal).
- Checkpoints are self-describing: `enrich.cfg` f32[5] = [rank, gelu,
  content_gate, a_mode, residual]. Legacy 4-slot cfgs load as residual
  `none`; M1-era files (w_q/w_k/w_v/w_o) are incompatible (expected to
  fail loading). Saves are atomic: serialized to `<path>.tmp` then renamed
  over the target (`Enricher::save`).
- Monitoring: log lines carry a `u` column (mean content-gate value, NaN
  unless V3); the collapse detector still watches the scalar gate — the
  pre-u tax keeps B alive even if u closes, so no separate u-abort.

Verified (all fresh variants): identity regression — baseline, fresh
V0/V1/V2/V3-d64 and PCA-r32 checkpoints produce bit-identical "Once upon a
time" greedy output; old-format (cfg-less) V0 loads; resume with a
mismatched variant bails with a clear error.

### Claim R: pick-orthogonal readout source (v2.2)

Question: would the readout benefit from seeing not the previous state `h`
but `h − alpha·e`, where `e` is the fed token's embedding — the thrown-away
mixture minus its picked share ("thought-residual")? The scale is forced,
not chosen: the only subtraction making an orthogonal decomposition is the
projection coefficient

```
alpha = <h, e> / ||e||^2        h' = h − alpha·e = (I − êê^T) h = P⊥ h
```

- **Normalization-invariant**: scaling `e` by `c` divides `alpha` by `c`;
  the residual vector is identical (row-norm spread absorbed exactly; no
  table-global normalization needed).
- **Interpretable**: on-manifold (||q|| = ||shat|| = sqrt(640)),
  alpha ≈ cos(state, pick) in [−1, 1] (Cauchy–Schwarz, no clamping) — the
  same aligned inner product the tied head evaluates for the logit,
  measured in the readout's own pre-final-norm geometry: the model's own
  confidence in the pick, by construction.
- **Degenerate tail**: ||h'||^2 = ||h||^2 (1 − alpha^2); as alpha^2 → 1
  the residual → 0 and the existing rmsnorm epsilon (+1e-6) turns it into
  a ~zero readout source ("nothing left to read") without division blowup
  (amplification < sqrt(H) even at alpha^2 = 1 − 1e-7). Near-degenerate
  rows (alpha^2 > 0.9999) are counted in the logs — a free collapse
  signature.
- Rejected scales: norm-matched (`||h||/||e||`) subtracts a full
  token-sized chunk regardless of alignment — content-free energy at
  cos ≈ 0, *added* energy at cos < 0 (the magnitude-without-content
  failure mode, M1 lesson); `e = E^T p` (expected mixture) is a valid
  sibling but costs a 262k×640 matvec per step (~335 MFLOPs) unless
  top-k truncated.

What it changes mechanically — a conditioning change, not a bottleneck:

- The residualization removes exactly one direction (the pick's: cos^2 of
  the energy); the low-dim hypothesis is untouched — what changes is *what
  the bottleneck summarizes*: belief-minus-pick instead of raw belief.
  The doc's motivating concept ("only a single discrete token id survives")
  becomes operational: the readout reads exactly what did not survive
  sampling.
- The module cannot learn this itself: A is static, the pick direction
  varies per step, and at teacher forcing the pick is the *target*
  (unrecoverable from the state) — a genuine capacity addition, not a
  reparam of W.
- The copy-the-pick shortcut (M1 run-5b) is removed from the *source*: the
  readout no longer must learn to ignore the pick.
- Confidence-adaptive by construction (Finding 7's "strength must be
  per-step", satisfied by the pick's own geometry): alpha ≈ 0 at flat
  steps (the variant *contains* the baseline exactly where it does not
  matter); at confident steps the source shrinks by cos^2.
- ICL-safe choice of `e`: subtracting the *current* pick q_t leaves the
  previous token's residual-stream persistence chunk intact (the copy
  handle the induction machinery reads); subtracting q_{t-1} instead
  removes the identity-carryover chunk — "state beyond what was just
  processed" vs "state beyond what was just predicted".

Gradient geometry: P⊥ is norm-non-expanding (symmetric, idempotent,
||P⊥|| = 1 — the contraction analysis and tau recipe are unchanged); `e`
stays a detached leaf (no new embedding-table gradients); P⊥ acts only on
the readout-recurrence credit path — the immediate path and the KV path
(the two dominant credit routes at init) are untouched. Central
falsifiable consequence: the chain's value function loses pick-alignment
shaping — credit to "make the state align with the pick" is exactly zero;
feature (kills echo-steering) or bug (kills confidence-shaping), decided
empirically.

Limits (stated precisely): source-orthogonality ≠ output-orthogonality —
B can still rotate complement content toward the pick direction, so a
learned echo-through-rotation remains possible (bounded by the tax); the
guarantee is on what the module *reads*, not what the model *receives*
(residualizing the injection output instead is rejected: it would
constrain what the model receives; the guards bound energy, never
direction). Training/deploy duality: alpha uses the fed token — forced at
training, sampled at deploy — the same duality q_t already has. Precedent:
the delta-rule / fast-weight-memory principle (operate on the deviation
from the current key, not the raw state), applied on the read side.

Variant config: `residual pick` reads `shat_r = rmsnorm(shat − alpha·q_t)`
(the gate keeps the raw [q; shat] view); `none` (default) is unchanged
v2.1. Parameter-free: +2 dots + 1 rms per step (~2k FLOPs vs ~540 MFLOPs
for the frozen model). The PCA Gram collector collects `sh_r`; new log
columns `alpha` (mean |alpha|) and `rem` (mean alpha^2; NaN unless pick).
Zero-init preserved trivially: at W/B = 0 the injection is 0 regardless of
the readout source.

## Training as TBPTT

Training == inference semantics: a **lockstep batch cascade** over prompt
+ answer with forced tokens. At each position t (batch B rows stepping
together): enrich the input from the previous state, run the frozen layers
over the growing KV prefix, push the state. At answer slots the fed token
is either the target (CE warmup) or **hard-sampled from the cascade's own
previous-step logits** (ST phase — no two-pass, no state-source mismatch).
The loss is transition-paired CE at `answer_start - 1 + k` vs `answer[k]`.

Gradient paths from an answer-slot CE back to the module params:

- **Immediate** (never truncated): loss -> 18 frozen layers -> x_t -> W,
  g_t of step t. Survives KV detaching via the residual stream and the
  query path.
- **KV path** (measured dominant at init): x_j -> K_j,V_j -> attention at
  later slots. NOT gated by W's magnitude: at W = 0 the state-chain is
  exactly zero and this is the *only* cross-step credit — how prompt-tail
  writes earn credit at the answer slots (the measured 5-8 attention
  window alone carried +20.4 of the 36.4 total W-gradient at init).
- **Chain**: dL/dshat_{t-1} -> step t-1's W/gate -> shat_{t-2} -> ... The
  one-step Jacobian is `ds_t/ds_{t-1} = J_f * (g W)` — exactly zero at
  W = 0 (the recurrence "wakes up" as W grows), contractive thereafter.

### Truncation: the tau-sweep

Rule: **tau such that c^tau < 0.01** on the state-chain path; the KV
path's horizon is data-dependent and must be measured. The tau-sweep
grad-share diagnostic (isolated per-process runs, batch[0] T=64, batch 8,
fresh W=0) measured it directly:

| config | grad norm (W) | peak RSS | verdict |
|---|---|---|---|
| tau=4, exact KV | 16.08 | 3.8 GB | best gradient-per-byte |
| **tau=6, exact KV** | **27.71** | **5.3 GB** | **the knee (default)** |
| tau=8, exact KV | 36.43 | 6.8 GB | +8.7 grad for +1.5 GB (diminishing) |
| tau=8, KV detached | 11.78 | 5.5 GB | strictly dominated by tau=4 exact |
| tau=16, KV detached | 11.78 | 10.1 GB | pointless: identical grad to tau=8 detached |

- KV detaching captures only 32% of the exact-KV gradient at the same
  horizon — at W=0 it is the only cross-step path.
- The detached-mode horizon saturates by tau=8 (tau=16 identical),
  confirming the c^tau analysis for the chain path.
- **tau=6 exact is the knee**: 72% of tau=8's gradient at 78% of its
  memory. The sweep used the shortest bucket (T=64); the training loop's
  longest bucket (~T=90) scales the in-graph KV cats ~1.4x — batch-8
  tau=8 exact would peak ~8 GB in-loop (OOM-hang on the 14 GB machine).
- Cross-step contraction: c = ||g W|| ||J_f|| <= inj_ratio * (||q|| /
  ||shat||) * ||J_f||. With the penalty holding injection at 0.25-1x
  input norm: c ~ 0.1-0.5 => c^8 <= 4e-3 — everything beyond ~8 steps is
  numerically dead. The M1 run-5b 131x takeover (c ~ 3, BPTT explosion)
  is what the pre-gate tax + monitors prevent.

**Default recipe: tau=6, kv_grad=true (exact), batch 4, accum 4.**

Options considered: (a) uniform TBPTT, exact KV — prefix positions
no-grad, last tau in-graph, transition CE (**chosen**, tau=6); (b)
KV-detached (`--train-kv-grad false`) — 32% of the exact gradient,
fallback for memory-starved setups; (c) hierarchical local losses (aux
next-token CE at every m-th position, frozen head, weight ~0.05-0.1) —
deferred, add only if prompt-side learning is starved: **triggered by
overnight run 2** (Results); (d) DEQ / implicit differentiation —
unnecessary (finite single-state recurrence, weakly contractive,
explicit).

Known limitation of (a) at tau=6: prompt-side enrichment older than ~5
positions before the answer gets zero credit (it still *runs* in forward —
fidelity preserved — it just does not learn). Inherent truncation trade;
(c) is the mitigation.

### Memory and time (measured)

Environment: 8-core CPU, 14 GB RAM, F32 safetensors backend only (GGUF
cannot train).

- The frozen embedding table and tied head were removed from the graph
  (detached lookup + surrogate head loss `sum(normed * g)` with
  `g = (softmax - onehot)/n @ W_head` detached): exact gradients, saves
  ~2 GB of head/table grad buffers per backward. Original estimates
  (~10-11 MB/step) were ~50x optimistic: measured backward peaks are
  dominated by in-graph KV cat/expand copies per layer-position (scaling
  ~tau x T x batch), frozen-weight sum-grads (~0.45 GB), and GradStore
  retention until backward completes.
- **Gradient accumulation** (`--train-batch B --train-accum M`): effective
  batch = B*M; in-graph memory scales with B only, while the cascade
  forward is weight-streaming-bound (batch 4 costs ~the same wall time as
  batch 8).

| recipe (tau=6 exact) | peak RSS | eff. batch | s/micro | s/eff.step |
|---|---|---|---|---|
| batch 8 x accum 1 | ~7.5-8 GB (longest bucket) | 8 | ~12 | ~12 (risky) |
| **batch 4 x accum 4 (default)** | **~3.5-4 GB** | **16** | **~12** | **~50** |
| batch 4 x accum 8 | ~3.5-4 GB | 32 | ~12 | ~100 |
| batch 4 x accum 16 | ~3.5-4 GB | 64 | ~12 | ~190 |

- Forward is memory-bound: 1.08 GB weight streaming per position-step,
  ~0.13-0.2 s (nearly batch-independent); ~8-11 s cascade forward
  (T=64-90) + 1.3-5 s backward => ~12 s/micro-step at batch 4-8.
  Per-process runs are the honest memory number — in-process sweeps
  ratchet the allocator high-water mark.
- `--train-steps-*` count **micro-steps**: the default 150+300 micros =
  112 effective steps ~ **1.6-1.9 h**; a full night: 600+1800 micros =
  600 effective steps ~ 8 h (~2.5-3k micros ~ 7-10 h).
- Gradient magnitudes: dL/dW entry ~ 0.02-0.15 per contributing (slot,
  step), ~ O(0.1-1) accumulated through the O(1)-O(2) frozen-layer
  Jacobian. AdamW normalizes scale; the real concerns are gradient noise
  (<= 32 loss tokens per batch — hence effective batch 16 halving noise
  vs batch 8) and cold-start magnitude (lr warmup + global-norm clip).

### Procedure and guards

Phases (each gates the next):

1. **Gradient sanity** (single example): cascade forward, loss decreases,
   grads only on `enrich.*`, per-tensor grad norms, tau-sweep, measured c.
2. **CE warmup** (~150 steps, forced targets): cold-start loss ~4.4 nats;
   the W-only wakeup stage needs stable states.
3. **Hard-sampled ST phase** (~300 steps): at each answer slot, with
   probability p_forced (annealed 1 -> 0 across the phase) feed the target
   token, else sample hard from softmax(logits_{t-1} / gt), gt annealed
   ~1.0 -> 0.1. Scheduled forcing (abrupt ST entry collapses the gate —
   M1 lesson) + on-policy states (exposure-bias kill for <= 4-token
   answers). CE on targets throughout.
4. **Evaluation**: teacher-forced EM (transition-paired argmax) +
   free-running greedy EM (cascade prefill) + guided-first variant
   (separates first-step bootstrap from destabilization) + bench transfer
   (digits/trivia unharmed) + the fresh-identity bench control (Results).
5. Optional later: aux local CE (option (c), triggered); elementwise
   gate; REINFORCE/GRPO (M3).

Kept guards: gate bias +2 open-start, lr warmup (50 steps), warmup-only
gate floor 0.5 with ST-phase anneal, gate-collapse detector (auto-abort +
forensic checkpoint), 5-min checkpoints, 2-3 runs per claim (candle CPU
matmuls are not run-to-run deterministic — thread scheduling changes
float summation order).

**Injection penalty (pre-gate)**: `loss + lambda * mean(||W shat||^2 /
||q||^2)` — the tax is on the pre-gate injection so closing the gate
cannot dodge it (M1 lesson (a)). Healthy band 0.25-1x relative injection
energy; also serves as BPTT well-conditioning (bounds c). Monitor both
pre-gate and applied (post-gate) ratios. lambda is d-dependent: 0.1 for
d=32 / V0 full-rank, 1.0 for factored variants at d >= 64 (calibration
verdict below).

**Watch list**: gate < 0.05 (auto-abort + forensic dump fires), pregate
> 1 (takeover watch), applied p/ans divergence, CE trend through the
forcing anneal, RSS in `top` (expect ~4-5 GB steady at batch 4; ~7 GB
measured on the overnights), and for residual runs alpha ~0.11-0.13 /
rem ~2% / degen 0.

## Results

### M1 (attention-pooling "thought pool"): lessons learned

The first architecture (K=8 attention-pooling w_q/w_k/w_v/w_o + gate,
r=192, answer slots only, two-pass parallel forward, CE warmup ->
ST-Gumbel) never beat baseline across six runs. Lessons carried into v2:

1. A capitalization bug (lowercase targets + case-sensitive scoring) made
   CE near-impossible and silently zeroed free-running EM — score exactly
   what you train on.
2. Gate dynamics need protection on both ends: open-start bias (+2.0), lr
   warmup, and a warmup-only gate floor keep the module alive;
   teacher-forced -> free-running transitions re-triggered collapse in
   every run without them.
3. Candle CPU matmuls are not run-to-run deterministic; report across 2-3
   runs; checkpoints are the safety net.
4. At extreme injection (131x / 0.1x), answer-slot enrichment produced
   zero teacher-forced gain: run 5b learned to *replace* the token
   identity, run 6b equilibrated at the identity. **(a) tax the pre-gate
   injection** (a post-gate tax is dodged by closing the gate);
   **(b) a magnitude penalty without a floor is escapable; a floor
   without a tax is a blindfold.**
5. The objective itself was wrong: `targets = answer` with CE pairing at
   `answer_start + i` consumes `answer[i]` as its input, so the loss was
   "re-emit the token just consumed" (echo maintenance), not next-token
   prediction — the correct transition pairing is `answer_start - 1 + i`
   (the state *before* consuming `answer[i]` predicts it). Signatures,
   all consistent with the run record: the near-copy solution is almost
   free for a tied-head LM (loss 17 -> 0.7-2.3 without answer knowledge);
   the free-running attractors ("KitMilMilMil...") are exactly the
   trained echo; TF-EM converged while FR-EM collapsed (TF measured echo
   accuracy, FR the never-trained transition). The ST phase had the same
   index shift. The v2 redesign fixes all three pairing sites (CE, ST
   sampling, state source).

### First v2 dynamics smoke (run v2-1): healthy on every axis

tau=4 exact, batch 4 x accum 1 (60 effective steps), lr ramp 10, gate
bias +2 / floor 0.5, inj_l2 0.1, seed 42, ~9.5 min. Cold start ~4.4 nats
(M1's 14-17 was the echo objective's inflation); the ST-entry bump (7.6)
declines (-> 5.4) while scheduled forcing anneals. No gate collapse
(0.94 -> 0.98 flat; the auto-abort never fired); pregate grows slowly
inside the lower healthy band (0 -> 0.245). Rising |g| during warmup
(17.7 -> 46) is the designed cross-step wakeup signature; declines at ST
(25). Prompt-side writes earn credit (applied p/ans ~ 0.23/0.35) — the
KV-path credit the M1 answer-only design could never get. tau=4 is
speed-smoke only; tau=6 (the default) covers 4-token answer buckets
exactly.

### v2.1 calibration smokes

d-sweep (V2/gelu at the v2-1 recipe, seeds 42/31337): probe pregate grew
0.40-0.47 (d32) < 0.85-1.29 (d64) < 1.44-1.54 (d128) — reproducibly
d-dependent (V0 grew slowest of all, 0.245, despite the largest init
gradient): the bottleneck converts gradient into injection *faster* —
fewer, better-aligned directions compose under Adam sooner. Consequence:
**lambda 0.1 (tuned on V0) under-brakes d >= 64**, which exits the
0.25-1x band within 60 steps in both seeds. 60-step probe CE could not
rank capacity (means 6.99 / 7.68 / 5.14 for d32/64/128, non-monotone —
gradient noise dominates at this horizon). Mechanics held everywhere:
gate 0.99 flat, no collapse, no abort, RSS ~5 GB, ~9.5 min per run.

Lambda calibration (same recipe, `--train-inj-l2` swept at fixed d; probe
= held-out forced CE after 60 effective steps; fresh baseline 4.274):

| d | seed | pregate lam 0.1 | 0.3 | 1.0 | probe CE 0.1 | 0.3 | 1.0 |
|---|------|------|-----|-----|------|-----|-----|
| 64 | 42 | 1.292 | 1.301 | **0.621** | 9.94 | 8.19 | 5.97 |
| 64 | 31337 | 0.850 | 0.707 | — | 5.41 | 5.08 | — |
| 128 | 42 | 1.543 | 0.967 | **0.586** | 4.87 | 4.68 | **3.32** |
| 128 | 31337 | 1.436 | 0.816 | — | 5.40 | 4.54 | — |

Verdict: **lambda = 1.0 for factored variants at d >= 64**; d=32 stays at
0.1 (its lambda-0.1 dynamics were already in band). lambda 1.0 *improved*
CE instead of crushing it (d128/s42 probe 3.32; in-train ST loss
declining through the anneal) — restrained injection looks
better-conditioned, not starved. Caveats: probe CE is teacher-forced
(~40 rows), 60-step horizon — free-running EM remains the verdict metric.
(The original baseline reading 5.09 predates the held-out dedup redraw.)

### Claim-R smokes

Protocol identical in shape to the calibration runs (matrix d64/d128 x
seed 42/31337, ± the residual flag). Fresh-init signature b 3.247 (pick)
vs 3.175 (none) — wakeup intact; fresh-pick generation bit-identical to
baseline; legacy 4-slot cfg loads; resume mismatch bails; fresh-pick
held-out probe ce == fresh-none ce (4.274). alpha stats: mean |alpha|
~ 0.11-0.13, mean removed fraction ~ 2% — stable across warmup/ST/probe,
both ranks, both seeds (a property of the frozen states, weakly affected
by the injection); no degenerate hits. Mechanics: gate 0.98-0.99 flat,
pregate in band, ~10.7 min/run (the residual op is free); the residual
mildly brakes injection growth (pick pregate <= none in 3 of 4 cells).

| d | seed | none pregate | pick pregate | none CE | pick CE | CE delta |
|---|------|------|------|------|------|------|
| 64 | 42 | 0.621 | 0.551 | 5.969 | 4.890 | **−1.08** |
| 64 | 31337 | 0.546 | 0.395 | 5.206 | 4.820 | **−0.39** |
| 128 | 42 | 0.586 | 0.384 | 3.319 | 4.168 | +0.85 |
| 128 | 31337 | 0.382 | 0.416 | 4.327 | 3.414 | −0.91 |

No Claim-R decision at the smoke horizon (teacher-forced probe CE, 60
steps, noise-ruled). **d64, pick vs none** is the only cell with a
consistent directional signal (−1.08, −0.39) and the tightest seed spread
— the overnight candidate; d128 is a coin flip (seed noise dominates;
only its residual mode carried as an exploratory cell).

### Claim-R overnights

Three runs at the overnight recipe (tau=6 exact KV, batch 4 x accum 4,
600 + 1800 micros = 600 effective steps ~ 8 h each, lr ramp 50 default,
lambda 1.0 per the calibration verdict): (1) d64 none control — doubles
as the never-run v2.1 d64 overnight; (2) d64 pick (the candidate); (3)
d128 pick, exploratory (dropped after the smokes: seed sign-flip).
Residual watch: alpha ~0.11-0.13 / rem ~2% (a drift is a
state-distribution warning, not a module failure); degen > 0 = collapse
signature.

**Run 2 (d64 pick)**: ~6.5 h, 74 checkpoints, zero aborts, atomic saves.
Mechanically clean: gate 0.94 -> 0.998 flat, alpha/rem/degen in band, RSS
~7.0 GB steady (watch item: above the 4-5 GB estimate). Pregate 0.03-0.06
— never entered the 0.25-1x band (the lambda-1.0 tax plus the ST lr held
injection small); applied p/ans ~0.05/0.4 — answer-slot-concentrated,
prompt-side starved (the KV-path credit did not materialize at this
scale). No ST-entry shock (p_forced anneals over 1800 micros).

| metric (100 held-out) | baseline | trained |
|---|---|---|
| teacher-forced EM | 37.9% | **65.8%** |
| free-running EM | 26.5% | 26.0% |
| FR, first-token-guided | 86.5% | **57.0%** |

Probe (trained, held-out): gate 0.998, pregate 0.085, applied 0.056/0.634,
ce 2.185 (fresh baseline 4.274) — the forced-mode signal is real.

Bench transfer (digits-20 n=100, trivia n=40; enricher mode = cascade
everywhere): digits-20 100% -> **0%** (p=0.000), trivia 72% -> **2%**
(p=0.000).

Findings: (1) the pick-readout learns a usable state summary under
forcing — TF-EM +28 points, probe CE halved; the smoke CE wins carry to
overnight scale. (2) Zero free-running transfer plus active
destabilization — guided-first drops 29.5 points: with a correct first
token the trained cascade derails continuations (name-blend attractors,
"Kitu"/"Milorp" echo texture); small per-step injections compound over
self-generated states (the TF-CE-flatters-over-injection regime, now at
tiny injection scale). (3) Off-task the module is catastrophic —
diagnosed below: the trained weights alone cause the collapse. (4) Aux
local CE (option (c)) triggered: its deferral condition was "add only if
prompt-side learning is starved" — applied p/ans 0.05/0.4 with prompt-side
flat all night is exactly that signal.

**Fresh-identity bench control** (a fresh identity enricher of the same
variant — rank 64, gelu, pick, seed 31337 — through the identical bench
invocation, digits-20 n=100): baseline 100%; fresh-identity 100%
(p=1.000, mean tokens 40.7 == 40.7); trained (run 2) 0%. The
cascade-prefill bench path is bit-exact at B=0 — **the trained weights
alone cause the off-task collapse** (and the guided-first
destabilization): the module only ever saw state-track / induce-d state
distributions; off-distribution its per-step injections are poison, and
the bench mode applies ~90 of them over the prompt before the answer.
Lesson: **the fresh-identity bench row is the cheap control that
separates "path damage" from "weight damage" — run it whenever an
enricher bench row collapses.**

**Run 1 (d64 none control)**: deviated split (300 warmup + 2100 ST, same
2400-micro total; a deliberate second variable, so pick-vs-none
attribution is confounded — accepted before launch). ~6.1 h train + eval,
0 aborts, mechanics clean (gate 0.995-0.998 flat, RSS ~7.0 GB). The none
readout injects far more aggressively than pick: pregate 0.23 by ST 0
(pick never left 0.03-0.06), applied answer-side climbing to 1.0-1.65x
input energy in the late anneal — the watch-list takeover line, crossed
on the applied side. The fully on-policy tail showed instability spikes
(st 2060-2070: CE 4.6, |g| 34-53) — the compounding loop visible in
training itself.

| metric (100 held-out) | baseline | none (pick) |
|---|---|---|
| teacher-forced EM | 37.9% | 59.1% (65.8%) |
| free-running EM | 26.5% | **15.5%** (26.0%) |
| FR, first-token-guided | 86.5% | **39.5%** (57.0%) |
| digits-20 bench | 100% | 0% (0%) |
| trivia bench | 72% | 5% (2%) |

Findings: (1) the damage is v2-recipe-wide, not Claim-R-specific — the
none control collapses off-task and destabilizes free-running, the same
failure class as pick, only worse; the residual treatment is exonerated
as the special culprit. (2) Pick was the milder failure on every
deployment metric while matching the TF gain (65.8 vs 59.1): the
pick-orthogonalization measurably brakes injection growth at scale (3-5x
less energy; applied never left ~0.5x vs none's 1.6x) — Claim R's
anti-echo mechanism works. (3) The split did not rescue free-running
(none@300+2100 worse than pick@600+1800): more on-policy exposure at the
current guards lets the injection grow into the takeover band instead of
stabilizing deployment; the residual x split confound is unresolved for
exact attribution, but the qualitative conclusion does not depend on it.
(4) The central blocker is the paradigm, not the variant: per-step
injection helps only the forced training tasks and poisons everything
else. Next levers: aux local CE for prompt-side credit; bench-informed
state coverage (the module never saw off-task state distributions); an
applied-energy cap given the gate never closes (0.998) — the M1
post-gate-tax objection assumed the gate could close, which it
demonstrably does not here.

## Tasks

Synthetic, seeded generators (reusing the bench machinery), held-out
split, baseline EM calibrated into the 30-80% band:

- `state-track`: 3 nonsense agents x nonsense objects; chain-of-custody
  statements with a distractor; answer = one holder name (1-2 tokens).
  Probes explicit state carrying through the recurrence.
- `induce-d`: 6 in-context nonsense-word -> number mappings + 2-3
  distractor statements; statements are SHUFFLED, so the queried
  (last-drawn) pair sits at a random position and the "copy the last
  stated number" shortcut fails; known 60% baseline band and known
  feedback sensitivity (Findings 6-7).

4k train + 200 held-out per task. Held-out prompts are deduplicated
against train (and within held-out; redraw on collision): the plain
sequential split left ~3% of held-out state-track prompts as literal
training prompts (the distinct-prompt space is only ~1e5 for state-track;
induce-d was already collision-free).

Bench validity: the original bench `induce` is an in-family validation of
induce-d, not independent transfer — its statements are NOT shuffled, so
it is solvable by echoing the last stated number. Use the binding-honest
variants `induce-sh` / `induce-early`; details in the bench-validity note
in FEEDBACK.md. digits/trivia remain clean transfer measures.

## Inference integration

- With `--enricher <path>`: prefill is replaced by the sequential forced
  cascade — one single-position forward per prompt token (forced ids,
  enricher active, state pushed each step). Cost ~0.13-0.2 s/position =>
  ~10-16 s per 80-token prompt (bench slows accordingly; acceptable for
  the suite).
- Decode loop unchanged in shape: each sampled token is embedded, enriched
  from the previous state, forwarded at the next position, state pushed.
- No enricher => all paths untouched (bit-identical regression battery).
- Sliding-window layers: inference uses the existing rotating caches;
  training assumes T < 512 (sliding window) so plain causal masks suffice
  for the task lengths (~60-90), asserted in code.

## Reproduce

```sh
# smoke recipe (~1.6-1.9 h: 150+300 micros = 112 effective steps)
cargo run --release -- --cpu --train-enricher --model-dir models/gemma-3-270m-it \
    --train-steps-warmup 150 --train-steps-st 300 \
    --train-out enricher_v2.safetensors

# full-night form (the Claim-R overnight recipe: 600+1800 micros ~ 8 h);
# lambda 1.0 applies to factored variants at d >= 64 (calibration verdict);
# d=32 / V0 uses the 0.1 default
cargo run --release -- --cpu --train-enricher --model-dir models/gemma-3-270m-it \
    --train-steps-warmup 600 --train-steps-st 1800 \
    --train-tau 6 --train-kv-grad --train-batch 4 --train-accum 4 \
    --train-gate-bias 2.0 --train-lr-warmup 50 --train-gate-floor 0.5 \
    --train-inj-l2 1.0 \
    --train-enrich-rank 64 --train-enrich-act gelu --train-enrich-residual pick \
    --train-out enricher_r_d64_pick.safetensors

# fresh variant checkpoint + gradient-signature sweep (no training)
cargo run --release -- --cpu --model-dir models/gemma-3-270m-it \
    --train-enricher --train-steps-warmup 0 --train-steps-st 0 \
    --train-skip-eval --train-sweep "4:1" --train-batch 2 \
    --train-enrich-rank 64 --train-enrich-act gelu \
    --train-out /tmp/enr_v2d64.safetensors

# PCA-calibrated A
cargo run --release -- --cpu --model-dir models/gemma-3-270m-it \
    --train-enricher --train-steps-warmup 0 --train-steps-st 0 \
    --train-skip-eval --train-sweep "4:1" --train-batch 2 \
    --train-enrich-rank 64 --train-enrich-a pca \
    --train-out /tmp/enr_pca64.safetensors

# tau-sweep diagnostic in isolation (per-process peak RSS is the honest
# number; in-process sweeps ratchet the allocator high-water mark)
cargo run --release -- --cpu --train-enricher --model-dir models/gemma-3-270m-it \
    --train-steps-warmup 0 --train-steps-st 0 --train-skip-eval \
    --train-sweep "6:1" --train-out /tmp/enr_sweep.safetensors

# inference with cascade prefill
cargo run --release -- --cpu --model-dir models/gemma-3-270m-it \
    --enricher enricher_v2.safetensors -p "..." -n 40 --temperature 0
```
