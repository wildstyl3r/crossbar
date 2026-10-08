# ENRICHER.md — trained state-readout enricher: design, BPTT analysis, plan

## Motivation and concept

The intuition behind the feedback approach: during one token processing step
many different "memories" (associations) are activated and interact to form
the output vector, but that rich mixture is thrown away at the sampling step
— only a single discrete token id survives, and the next step's first layers
(almost) start from scratch. Later layers retain access to the emergent
mixture via the KV cache / attention, but early layers see only one token
embedding. The looped / universal-transformer line of work suggests that
spending more compute on the same, already-picked concepts (thinking on the
same tokens for longer) improves performance. The aim: **spend more compute
on the concepts picked during the previous token processing, so that ideas
that started to emerge are not lost again and again.**

Findings 6-8 (FEEDBACK.md) sharpened this:

- The expected-embedding feedback damages in-context induction
  (60% -> 43%): the induction head needs the *exact* previous-token identity
  as its attention key; the expected embedding smears that key over the
  token's neighborhood (recall survives, precision dies: "68" -> "6").
- Blending with the sampled token (a=0.5) rescues most of the damage
  (-> 56%) and is uniformly non-harmful — the sampled token re-anchors the
  attention key.
- The iterated hand-set prefill cascade at gentle strength (K=2, a_p=0.25)
  is the only prompt-side mechanism that *improves* ICL (induce 78% vs 60%
  baseline, p=0.006).
- Hand-crafted feedback at fixed strength is uniformly destructive at
  strong settings (Finding 7): strength and content must be **learned,
  per step** — not hand-set.

This document specifies the redesigned trained mechanism: a minimal
**gated state readout** trained as an RNN with truncated BPTT, deployed as
a sequential cascade over both prompt and answer.

## M1 post-mortem (runs 1-6, 2026-10-05/06)

The first architecture (M1) was a K=8 attention-pooling "thought pool"
(w_q/w_k/w_v/w_o + gate, r=192) applied to answer slots only, trained by a
two-pass parallel forward (Pass-A states + Pass-B enriched re-forward) with
CE warmup -> ST-Gumbel. Terminal verdict across six runs:

1. The capitalization bug had two layers: lowercase targets made CE
   near-impossible (~17-25 nats) and the gate learned the escape hatch
   (switch itself off, zeroing the W_o gradient); case-sensitive scoring
   silently zeroed all free-running EM. Both fixed (capitalized targets,
   case-insensitive scoring).
2. Gate dynamics need protection on both ends: open-start bias (+2.0, sigmoid
   ~0.88), lr warmup, and a warmup-only gate floor keep the module alive;
   teacher-forced -> free-running transitions re-triggered collapse in every
   run without them.
3. candle CPU matmuls are not run-to-run deterministic (thread scheduling
   changes float summation order); fixed seeds do not guarantee
   reproducible trajectories. Report across 2-3 runs; checkpoints are the
   safety net.
4. **Runs 5-6 terminal verdict**: at 131x and 0.1x injection, answer-slot
   enrichment from last-8 Pass-A states produced **zero** teacher-forced
   gain. Run 5b learned to *replace* the token identity with
   W_o(pooled(states)) (~131x the input norm; decoding destroyed — the
   Finding-6 mechanism, now trained-in); run 6b (gate floor + penalty)
   equilibrated at the **identity** (bit-exact baseline at inference).
   Incentive lessons carried forward: **(a) tax the pre-gate injection**
   (a post-gate tax is dodged by closing the gate), **(b) a magnitude
   penalty without a floor is escapable; a floor without a tax is a
   blindfold.**

## Found defect in M1's objective: the CE pairing was self-echo (off-by-one)

Reading the M1 training code closely: `input_ids = [prompt, answer]`,
`targets = answer`, and the CE paired **logits at position
`answer_start + i`** with `targets[i] = answer[i]`. But position
`answer_start + i` *consumes* `answer[i]` as its input — the pairing asked
the model to re-emit the token it had just consumed, not to predict the
next one. The correct transition pairing is **logits at
`answer_start - 1 + i` vs `targets[i]`** (the state *before* consuming
`answer[i]` predicts it).

Consequences, all consistent with the run record:

- The trained objective was "echo maintenance", not answer production;
  the near-copy solution is almost free for a tied-head LM (loss 17 ->
  0.7-2.3 without any answer knowledge).
- The free-running attractors ("KitMilMilMil...") are exactly the trained
  echo behavior — training and deployment disagreed on the recurrence.
- Teacher-forced EM converged while free-running EM collapsed: TF-EM
  measured echo accuracy, FR-EM measured the real (never-trained)
  transition.
- The ST phase sampled the token fed at slot `i` from the logits at slot
  `i` (Pass-A) rather than from slot `i-1` — the same index shift.

The redesign fixes all three: transition-paired CE, sampling from the
cascade's own previous-step logits, states from the enriched cascade.

## Redesigned architecture: gated state readout (v2)

```
q_t     = E[a_t] * sqrt(h)                # sampled/forced token, on-manifold, ||q|| ~ 25
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
  (pre-final-norm last-position hidden) — the K=8 ring buffer is dropped.
- RMS-normalization makes the readout scale-free (better-conditioned
  gradients; the inj penalty stays meaningful as state norms drift).
- Single wakeup stage: at W=0, dL/dW != 0 immediately, but dL/dgate = 0
  until W moves; the gate only starts learning after the injection exists.
- Decisions (agreed 2026-10-07): scalar gate; RMS-normalized state; full
  cascade everywhere (training AND inference; parallel prefill is *never*
  used when the enricher is attached); no auxiliary losses in v0; tau = 8.

## Enricher variants v2.1 (2026-10-08): factored rank-d readout, nonlinearity, content gating

The v2 readout puts 409,600 of its 410,881 params into the full-rank W,
against <= 32 loss tokens per effective batch — the per-microbatch dL/dW is
a sum of ~O(50-100) rank-1 outer products, so most of W never earns signal
(weight decay is all that happens in its null space). The factored form
concentrates the same learning into 2*d*H entries and doubles as a direct
test of the module's core hypothesis (the thrown-away "emergent mixture" is
a low-dim summary). Compute is a non-issue either way: the enricher is
~0.3-3.3 MFLOPs/step vs ~540 MFLOPs + 1.08 GB weight-streaming for the
frozen model — wall time and the tau=6 recipe are unchanged, so variant
comparisons cost nothing extra per run.

Variants (selected by `--train-enrich-rank d`, `--train-enrich-act`,
`--train-enrich-content-gate`, `--train-enrich-a`):

```
V0 (default, rank 0):  inj = W shat                        W in R^{640x640}   410,881 params
V1 (rank d):           inj = B A shat                      A [d,H], B [H,d]   d=64: 83,201
V2 (V1 + act):         inj = B gelu(A shat)                same params
V3 (V2 + content gate):u = sigmoid(q Wu + u_z*z + b_u);    + 45,120 (d=64)
                       inj = g * B(u * z)                  128,321 total
```

- The scalar gate `g = sigmoid(w_g [q; shat] + b)` is unchanged in all
  variants — it already conditions on the incoming token; V3 adds
  *content* selection (which bottleneck features to inject for this token)
  on top of V2's *magnitude* selection.
- Init/wakeup identical in every variant: B (or W) zero-init => bit-exact
  baseline; dL/dB != 0 at init, dL/dA = dL/du = dL/dg = 0 until B moves
  (measured fresh-init sweep signatures, batch 2, tau=4: V0 w 16.38,
  V1-d64 b 5.06, V2-d64 b 3.18, V3-d64 b 2.80, PCA-r32 b 16.32 — a/u/gate
  all 0.000 as designed).
- Guards port unchanged: the injection penalty measures the pre-gate,
  pre-u energy `||B phi(A shat)||^2 / ||q||^2` (M1 lesson (a) — u cannot
  dodge it; one extra [B,d]x[d,H] matmul per step when V3 is on).
  Contraction picks up a <= 1.1 factor from gelu' (c^6 headroom intact).
- A treatment (`--train-enrich-a`): `trainable` (default; deterministic
  per-seed rows ~ N(0, 1/H), ~unit rows), `fixed` (ELM/reservoir reading,
  only B + gates train), `pca` (A set once from the top-d eigenvectors of
  the collected states' Gram matrix, then fixed — see below).
- PCA calibration (`pca`, fresh runs only): one no-grad cascade over the
  first 16 training batches (~2k state rows at batch 2), top-d
  orthonormalized power iteration (60 rounds) on the Gram matrix => A rows
  unit-norm (z coordinates ~unit variance, GELU-healthy). Idempotent on
  resume (deterministic given the seeded data order), ~2-3 min one-time.
  PCA-A reads state energy-aligned directions (fresh-init dB 16.3 vs 5.1
  for random A) — the prior pays off immediately in signal.
- Checkpoints are self-describing: `enrich.cfg` f32[4] = [rank, gelu,
  content_gate, a_mode]. v2 checkpoints (enrich.w family, no cfg) load as
  V0; M1-era files (w_q/w_k/w_v/w_o) are incompatible (expected).
- Monitoring: the log lines gained a `u` column (mean content-gate value,
  NaN unless V3). The collapse detector still watches the scalar gate; the
  pre-u tax keeps B alive even if u closes, so no separate u-abort in v0.

Verified (2026-10-08, all fresh variants): identity regression — baseline,
fresh V0, fresh V1/V2/V3-d64 and PCA-r32 checkpoints produce bit-identical
"Once upon a time" greedy output; old-format (cfg-less) V0 loads; resume
with a mismatched variant bails with a clear error.

### Variant experiment ladder (claims, 2-3 runs each)

1. **d-sweep (V1 linear)**: rank 64 vs 128 (vs 32 if EM-gain is flat) at
   the standard smoke recipe — if EM-gain grows monotonically to d=256+,
   the low-dim-mixture hypothesis is falsified; if d=32-64 saturates, the
   factored form is the better default.
2. **V2 (bundle)**: same d with `--train-enrich-act gelu` — free capacity
   widening; compare against the V1 run at equal d.
3. **V3 (content gate)**: separate claim after V1/V2 verdicts; watch the
   u column (open-start ~0.88 via u_b = +2), pregate vs applied divergence.
4. Calibration note: the KV path carries ~68% of the cross-step gradient
   (tau-sweep) and is untouched by the paramization — expect modest EM
   deltas from architecture alone; the ladder's main readout is gradient
   health (grad norms, pregate band, CE trend) + held-out EM.

```sh
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

# overnight variant run (recipe identical to v2; tau=6 exact, batch 4 x accum 4)
cargo run --release -- --cpu --model-dir models/gemma-3-270m-it \
    --train-enricher --train-steps-warmup 600 --train-steps-st 1800 \
    --train-tau 6 --train-kv-grad --train-batch 4 --train-accum 4 \
    --train-enrich-rank 64 --train-enrich-act gelu \
    --train-out enricher_v21_r64.safetensors
```

### d-sweep smoke (2026-10-08): healthy mechanics, d-dependent injection growth

V2 (`--train-enrich-act gelu`) at the v2-1 recipe (tau=4 exact, batch 4 x
accum 1, 30+30 micros = 60 effective steps, lr ramp 10, lambda 0.1), seeds
42 / 31337, held-out probe after training (forced-token cascade CE; the
v2-1 V0 reference on the same protocol: pregate 0.245, applied 0.23/0.35,
ce 5.14):

| d | seed | pregate | applied p/ans | probe CE | st20 \|g\| |
|---|------|---------|---------------|----------|------|
| 32 | 42 | 0.396 | 0.39/0.53 | 6.72 | 10.5 |
| 32 | 31337 | 0.469 | 0.46/0.55 | 7.26 | 16.2 |
| 64 | 42 | **1.292** | 1.26/1.72 | 9.94 | 18.7 |
| 64 | 31337 | **0.850** | 0.83/1.05 | 5.41 | 24.8 |
| 128 | 42 | **1.543** | 1.49/1.95 | 4.87 | 19.9 |
| 128 | 31337 | **1.436** | 1.41/1.68 | 5.40 | 18.5 |

Findings:

1. **Injection growth rate scales with d, reproducibly**: probe pregate
   0.40-0.47 (d32) < 0.85-1.29 (d64) < 1.44-1.54 (d128); V0 grew slowest
   of all (0.245) despite the largest init gradient (16.4 vs 3.2 for d64's
   B). The bottleneck converts gradient into injection *faster* — fewer,
   better-aligned directions compose under Adam sooner.
2. **The tax calibration is d-dependent**: lambda 0.1 was tuned on V0's
   growth rate and under-brakes d >= 64, which exits the 0.25-1x band
   within 60 steps in both seeds. No aborts fired (the band check is
   monitoring-only); add a lambda 0.2-0.3 recalibration for d >= 64 or an
   explicit pregate-band warning/abort in the training loop.
3. **60-step probe CE cannot rank capacity**: means 6.99 / 7.68 / 5.14
   (d32/d64/d128), non-monotone with huge seed spread (d64: 9.94 vs
   5.41). Gradient noise (<= 32 loss tokens) dominates at this horizon.
   d128-s42 even beat the fresh baseline CE (5.09) — but at 1.5-1.9x
   applied injection, the regime where TF-CE gains flatter over-injection
   and need free-running EM to confirm (the M1 lesson).
4. **Mechanics held everywhere**: gate 0.99 flat, no collapse, no abort,
   RSS ~5 GB (batch 4, tau=4), ~9.5 min per run + ~1 min probe.

Next-run consequences: overnight capacity runs (600+1800, tau=6 exact,
batch 4 x accum 4) should use d=32 lambda=0.1 as-is, and d=64/128 only
after a 10-min lambda=0.3 calibration smoke confirms pregate <= 1x at 60
steps. Free-running EM + bench transfer, not probe CE, are the verdict
metrics.

### Lambda calibration smokes (2026-10-08): lambda 1.0 is the factored-variant brake

Same recipe, `--train-inj-l2` swept at fixed d (probe = held-out forced
CE after 60 effective steps; baseline ~5.09):

| d | seed | pregate lam 0.1 | 0.3 | 1.0 | probe CE 0.1 | 0.3 | 1.0 |
|---|------|------|-----|-----|------|-----|-----|
| 64 | 42 | 1.292 | 1.301 | **0.621** | 9.94 | 8.19 | 5.97 |
| 64 | 31337 | 0.850 | 0.707 | — | 5.41 | 5.08 | — |
| 128 | 42 | 1.543 | 0.967 | **0.586** | 4.87 | 4.68 | **3.32** |
| 128 | 31337 | 1.436 | 0.816 | — | 5.40 | 4.54 | — |

Findings:

1. **lambda 0.3 barely brakes d64** (1.292 -> 1.301 on s42) — consistent
   with AdamW's per-entry normalization: when the tax gradient is
   sub-dominant in magnitude *and* rarely flips an entry's sign balance,
   scaling it 3x changes little. lambda 1.0 does bite (0.62, in band).
2. **lambda 1.0 improved CE instead of crushing it**: d128/s42 probe CE
   3.32 (baseline 5.09), in-train ST loss *declines* through the forcing
   anneal (st20 4.85 vs 5.81 at lambda 0.1 and 7.3-8.9 for the rest of
   the matrix). Restrained injection looks better-conditioned, not
   starved. Caveats stand: probe CE is teacher-forced (~40 rows), 60-step
   horizon — free-running EM remains the verdict.
3. Calibration verdict: **lambda = 1.0 for the factored variants at
   d >= 64**; d=32 stays at lambda 0.1 (its lambda-0.1 dynamics were
   already in band; lambda 1.0 untested at d32).

## Training as RNN BPTT

Training == inference semantics: a **lockstep batch cascade** over
prompt + answer with forced tokens. At each position t (batch B rows
stepping together): enrich the input from the previous state, run the
frozen layers over the growing KV prefix, push the state. At answer slots
the fed token is either the target (CE warmup) or **hard-sampled from the
cascade's own logits at step t-1** (ST phase — no Pass-A, no two-pass, no
state-source mismatch). The loss is transition-paired CE at
`answer_start - 1 + k` vs `answer[k]`.

Gradient paths from an answer-slot CE back to the module params:

- **Immediate** (never truncated): loss -> 18 frozen layers -> x_t ->
  W, g_t of step t. Survives KV detaching via the residual stream and the
  query path.
- **KV path** (measured dominant at init): x_j -> K_j,V_j -> attention at
  later slots. NOT gated by W's magnitude: at W = 0 the state-chain below is
  exactly zero (dx/ds = gW = 0) and this is the *only* cross-step credit —
  it is how prompt-tail writes earn credit at the answer slots (the answer
  attention reads the recent prompt positions strongly; the measured 5-8
  window alone carried +20.4 of the 36.4 total W-gradient at init).
- **Chain**: dL/dshat_{t-1} -> step t-1's W/gate -> shat_{t-2} -> ...
  The one-step Jacobian is `ds_t/ds_{t-1} = J_f * (g W)` — exactly zero at
  W = 0 (the recurrence "wakes up" as W grows), contractive thereafter.
- At W=0 the recurrence is *broken* (c = 0, no chain gradient): the BPTT
  horizon grows only as fast as the learned injection. Early training is
  immediate + KV; cross-step chain shaping emerges as W grows.

## Gradient size estimates (batch 8 lockstep, T~80, F32 CPU)

**Memory (graph).** Original estimates (~10-11 MB/step, tau=8 ~ 90 MB)
were ~50x optimistic: the measured backward peaks (tau-sweep table below)
are dominated by (a) the frozen embedding-table and tied-head grad buffers
(~2 GB, since removed from the graph via the detached lookup + surrogate
head loss), (b) the in-graph KV cat/expand copies per layer-position
(scaling ~tau x T x batch), (c) frozen-weight sum-grads (~0.45 GB) and
candle's GradStore retaining all intermediate grads until the backward
completes. Measured peaks: tau=4/6/8 exact = 3.8/5.3/6.8 GB (batch 8,
T=64); the training loop's longest bucket scales ~1.4x. Per-process runs
are the honest number — in-process sweeps ratchet the allocator
high-water mark.

**Time.** Forward is memory-bound (1.08 GB weight streaming per
position-step, ~0.13-0.2 s; nearly batch-independent). Measured: ~8-11 s
cascade forward (T=64-90) + 1.3-5 s backward => ~12 s/micro-step at
batch 4-8. Wall time per *effective* step = accum x that; see the recipes
table in the TBPTT section. The default 150+300 micros ~ 1.6-1.9 h.

**Magnitude.**

- dL/dlogits = softmax - onehot; tied head: dL/ds ~ ||E^T(p - y)|| ~ 0.3-1
  at healthy loss (1-3 nats), ~1-1.5 at cold start (14-17 nats).
- Through the 18 frozen layers (residual-dominated, ||J_f|| ~ O(1)-O(2)):
  dL/dx_t ~ O(1).
- dL/dW entry ~ dx[d] * g * shat[p] ~ 0.02-0.15 per contributing
  (slot, step); accumulated over <= 4 slots x tau steps ~ O(0.1-1).
  AdamW normalizes scale; the real concerns are gradient *noise* (<= 32
  loss tokens per batch) and cold-start magnitude — lr warmup + global-norm
  clip.
- **Cross-step contraction** c = ||g W|| ||J_f|| <= inj_ratio *
  (||q|| / ||shat||) * ||J_f||. With the penalty holding injection at
  0.25-1x input norm: ||W|| <= 1, realistically c ~ 0.1-0.5 =>
  c^8 <= 4e-3 — everything beyond ~8 steps is numerically dead.
  The run-5b 131x takeover (c ~ 3, BPTT explosion) is exactly what the
  pre-gate tax + monitors prevent.

## TBPTT truncation strategy

Rule: **tau such that c^tau < 0.01** applies to the *state-chain* path; the
KV path's horizon is governed by attention weights (data-dependent, large
for recent positions) and must be measured. The **tau-sweep grad-share
diagnostic** (isolated per-process runs, batch[0] T=64, batch 8, fresh W=0,
2026-10-07) measured it directly — W-gradient norm and true peak RSS per
config:

| config | grad norm (W) | peak RSS | verdict |
|---|---|---|---|
| tau=4, exact KV | 16.08 | 3.8 GB | best gradient-per-byte |
| **tau=6, exact KV** | **27.71** | **5.3 GB** | **the knee (default)** |
| tau=8, exact KV | 36.43 | 6.8 GB | +8.7 grad for +1.5 GB (diminishing) |
| tau=8, KV detached | 11.78 | 5.5 GB | strictly dominated by tau=4 exact |
| tau=16, KV detached | 11.78 | 10.1 GB | pointless: identical grad to tau=8 detached |

Findings:

- **KV detaching captures only 32%** of the exact-KV gradient at the same
  horizon (11.78/36.43) — the doc's original "KV skip is the weakest path"
  assumption was wrong; at W=0 it is the *only* cross-step path.
- **The detached-mode horizon saturates by tau=8** (tau=16 identical):
  without the KV path the state-chain alone dies fast, confirming the
  c^tau analysis — and making tau=16-detached pointless.
- Marginal gradient-per-GB decreases (7.75/GB for 4->6, 5.81/GB for 6->8):
  **tau=6 exact is the knee** — 72% of tau=8's gradient at 78% of its
  memory.
- The sweep used the shortest bucket (T=64); the training loop's longest
  bucket (~T=90) scales the in-graph KV cats ~1.4x, so batch-8 tau=8 exact
  would peak ~8 GB in-loop — the OOM-hang regime on this 14 GB machine.

**Default recipe: tau=6, kv_grad=true (exact), batch 4, accum 4** — see
below for why accumulation, not batch 8.

Memory levers (measured 2026-10-07, per-process peaks):

- The frozen embedding table and tied head were removed from the graph
  (detached lookup + surrogate head loss `sum(normed * g)` with
  `g = (softmax - onehot)/n @ W_head` detached): exact gradients, saves
  ~2 GB of head/table grad buffers per backward.
- **Gradient accumulation** (`--train-batch B --train-accum M`): effective
  batch = B*M; the in-graph memory scales with B only, while the cascade
  forward is weight-streaming-bound (batch 4 costs ~the same wall time as
  batch 8). Wall time per effective step ~ M * (fwd + bwd):

  | recipe (tau=6 exact) | peak RSS | eff. batch | s/micro | s/eff.step |
  |---|---|---|---|---|
  | batch 8 x accum 1 | ~7.5-8 GB (longest bucket) | 8 | ~12 | ~12 (risky) |
  | **batch 4 x accum 4 (default)** | **~3.5-4 GB** | **16** | **~12** | **~50** |
  | batch 4 x accum 8 | ~3.5-4 GB | 32 | ~12 | ~100 |
  | batch 4 x accum 16 | ~3.5-4 GB | 64 | ~12 | ~190 |

  Effective batch 16 also halves the gradient noise vs batch 8 (the
  per-batch signal is <= 32 loss tokens); accum 8/16 trade wall time for
  noise — use only if noise demonstrably dominates. NOTE: `--train-steps-*`
  count **micro-steps**; the default 150+300 micros = 112 effective steps ~
  **1.6-1.9 h**. For a full night at the default recipe scale up (e.g.
  600 warmup + 1800 ST micros = 600 effective steps ~ 8 h).

Options considered:

- **(a) Uniform TBPTT, exact KV** — prefix positions no-grad (detached
  enriched input + detached embedding), last tau positions in-graph,
  transition CE. **Chosen (tau=6).**
- **(b) KV-detached TBPTT** (`--train-kv-grad false`) — keeps only
  immediate + chain. Measured at 32% of the exact gradient; fallback for
  memory-starved setups.
- **(c) Hierarchical local losses** — aux next-token CE at every m-th
  position (frozen head, weight ~0.05-0.1): gives prompt-side injections
  local credit beyond the tau horizon. Deferred (agreed): add only if the
  cross-step grad share shows prompt-side learning is starved.
- **(d) DEQ / implicit differentiation** — unnecessary: finite
  single-state recurrence, weakly contractive, explicit. Equilibrium
  machinery pays off only for iterate-to-fixed-point schemes.

Known limitation of (a) at tau=6: prompt-side enrichment older than ~5
positions before the answer gets zero credit (it still *runs* in forward —
fidelity preserved — it just does not learn). That is the inherent
truncation trade; (c) is the mitigation.

## Training procedure

Phases (each gates the next):

1. **Gradient sanity** (single example): cascade forward, loss decreases,
   grads only on `enrich.*`, per-tensor grad norms, tau-sweep, measured c.
2. **CE warmup** (~150 steps, forced targets): still warranted — cold-start
   loss 14-17 nats and the W-only wakeup stage needs stable states.
3. **Hard-sampled ST phase** (~300 steps): at each answer slot, with
   probability p_forced (annealed 1 -> 0 across the phase) feed the target
   token, else sample hard from softmax(logits_{t-1} / gt), gt annealed
   ~1.0 -> 0.1. Scheduled forcing (the run-4 lesson: abrupt ST entry
   collapses the gate) + on-policy states (exposure-bias kill for <= 4-token
   answers). CE on targets throughout.
4. **Evaluation**: teacher-forced EM (transition-paired argmax) +
   free-running greedy EM (cascade prefill) + guided-first variant
   (separates first-step bootstrap from destabilization) + bench transfer
   (digits/trivia unharmed).
5. Optional later: aux local CE; elementwise gate; REINFORCE/GRPO (M3).

Kept guards: gate bias +2 open-start (default), lr warmup (50 steps),
warmup-only gate floor 0.5 with ST-phase anneal, gate-collapse detector
(auto-abort + forensic checkpoint), 30-min checkpoints, 2-3 runs per claim
(CPU non-determinism).

**Injection penalty (pre-gate)**: `loss + lambda * mean(||W shat||^2 /
||q||^2)`, lambda = 0.1 — the M1 lesson (a): the tax is on the pre-gate
injection so closing the gate cannot dodge it. Healthy band 0.25-1x
relative injection energy; also serves as the BPTT well-conditioning
(bounds c). Monitor both pre-gate and applied (post-gate) ratios.

## Run v2-1 (2026-10-07): first dynamics smoke — healthy on every axis

Config: tau=4 exact, batch 4 x accum 1 (60 real optimizer steps), warmup 30
+ ST 30 micros, lr ramp 10, gate bias +2 / floor 0.5, inj_l2 0.1, seed 42,
~9.5 min wall, fresh W=0 start.

Trajectory (warmup -> ST):

| metric | warmup 0 | warmup 10 | warmup 20 | st 0 | st 10 | st 20 |
|---|---|---|---|---|---|---|
| ce | 4.42 | 3.78 | 4.11 | 7.58 | 5.96 | 5.40 |
| gate | 0.940 | 0.949 | 0.974 | 0.980 | 0.980 | 0.982 |
| pregate | 0.000 | 0.008 | 0.056 | 0.116 | 0.163 | 0.213 |
| applied p/ans | 0/0 | — | — | 0.132/0.281 | 0.169/0.271 | 0.215/0.349 |
| \|g\| | 17.7 | 33.4 | 46.0 | 40.4 | 40.8 | 25.3 |

Probe (held-out, fresh pass, after training): gate 0.987 | pregate 0.245 |
applied 0.233/0.353 | ce 5.14.

Readings:

1. **Cold start ~4.4 nats, not M1's 17**: at W=0 the transition-paired CE is
   the frozen model's natural answer uncertainty; the M1 echo objective had
   inflated it. ST entry bump (7.6) is the on-policy sampled tokens — it
   *declines* (-> 5.4) while scheduled forcing anneals 1 -> 0.33.
2. **No gate collapse anywhere** (0.94 -> 0.98 flat): the run-2/run-4
   collapse pattern did not reproduce with hard-sampled tokens + scheduled
   forcing + open-start bias + floor. The auto-abort never fired.
3. **No injection takeover**: pregate grows monotonically-slowly (0 ->
   0.245 over 60 steps) inside the lower healthy band; M1's failure was
   131x — here ~0.25x and the pre-gate tax is active.
4. **Rising \|g\| during warmup (17.7 -> 46) is the designed cross-step
   wakeup signature**: at W=0 only the immediate path exists; as W grows
   the state-chain turns on and adds gradient mass. \|g\| then declines at
   ST (25) as the loss stabilizes.
5. **Prompt-side writes earn credit**: applied p/ans ~ 0.23/0.35 — writes
   concentrate at answer slots (~1.5x) but prompt positions are far from
   zero. This is the KV-path credit the M1 answer-only design could never
   get.
6. **tau=4 is too short for long-answer buckets**: the
   `tau=4 < answer_len+1=5` warning fired on buckets with 4-token answers
   (the first transition's immediate gradient falls outside the window).
   tau=6 (the default) covers them exactly — tau=4 is speed-smoke only.

## Inference integration

- With `--enricher <path>`: **prefill is replaced by the sequential forced
  cascade** — one single-position forward per prompt token (forced ids,
  enricher active, state pushed each step). Parallel prefill is never used.
  Cost ~0.13-0.2 s/position => ~10-16 s per 80-token prompt (bench slows
  accordingly; acceptable for the suite).
- Decode loop unchanged in shape: each sampled token is embedded, enriched
  from the previous state, forwarded at the next position, state pushed.
- No enricher => all paths untouched (bit-identical regression battery).
- Sliding-window layers: inference uses the existing rotating caches;
  training assumes T < 512 (sliding window) so plain causal masks suffice
  for the task lengths (~60-90), asserted in code.

## Tasks (unchanged from M1)

Synthetic, seeded generators (reusing the bench machinery), held-out split,
baseline EM calibrated into the 30-80% band:

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

Bench relation (corrected 2026-10-08): the original bench `induce` is an
IN-FAMILY validation of the induce-d skill, not an independent transfer
measure — same statement/query format, 15/16 vocab overlap with the
training pool, and ~8x coverage of every (word -> number) pair across
training prompts (unexploitable: the mapping is random per example, so no
consistent association exists to memorize; the answer always derives from
the prompt's own context). Moreover its statements are NOT shuffled, so
the queried pair is always the last stated one — solvable by echoing the
last number with zero word->number binding (verified 400/400 rows in
bench_results.csv; see the bench-validity note in FEEDBACK.md). The
binding-honest bench variants are `induce-sh` (statements shuffled) and
`induce-early` (query drawn from positions 0..4, never last-drawn;
statements shuffled). digits/trivia remain clean transfer measures.

## Compute reality and execution plan

Environment: 8-core CPU, 14 GB RAM, F32 safetensors backend only (GGUF
cannot train). Default recipe **tau=6 exact, batch 4, accum 4** (effective
batch 16): ~12 s/micro-step, ~3.5-4 GB peak. The default 150+300
micro-steps run ~1.6-1.9 h (112 effective steps); scale `--train-steps-*`
for a full night (micro-steps ~2.5-3k ~ 8-10 h). Unattended runs with
5-min checkpoints (user-approved).

1. ~~M1 module + runs 1-6~~ (done; post-mortem above).
2. **This redesign**: enrich.rs v2 + cascade inference path + cascade
   training (TBPTT tau=6 exact) + transition-paired CE/eval + regression
   battery (done: no-enricher == fresh-enricher bit-identical, F32 + GGUF).
3. ~~Gradient sanity + tau-sweep~~ (done 2026-10-07: table above; tau=6
   exact chosen; embedding/head removed from the graph; accumulation
   implemented; per-tensor norms labeled + foreign-grad check).
4. ~~Data calibration smoke~~ (superseded by run v2-1: baseline CE ~4.4
   nats, held-out 5.1; the old echo-pairing numbers are obsolete).
5. ~~Short CE + ST run~~ (done: run v2-1 above — healthy dynamics, no
   collapse, no takeover; injection at the lower band edge after 60 steps).
6. Overnight full run (scale --train-steps-* to the night budget;
   checkpoints every ~5 min).
7. Evaluation: bench `enricher` mode (cascade) vs baseline; held-out EM;
   findings into FEEDBACK.md; commit.

### Next-run suggestions

- **Recommended overnight** (tau=6 exact, batch 4 x accum 4, ~8 h, 600
  effective steps):
  ```sh
  cargo run --release -- --cpu --train-enricher \
      --model-dir models/gemma-3-270m-it \
      --train-steps-warmup 600 --train-steps-st 1800 \
      --train-tau 6 --train-kv-grad --train-batch 4 --train-accum 4 \
      --train-out enricher_v2.safetensors
  ```
  (lr ramp 50 default; gate-bias/floor/inj defaults already match.)
- **Medium smoke first** (~40 min, optional): same but
  `--train-steps-warmup 150 --train-steps-st 150 --train-accum 1` — verify
  injection climbs through the 0.25-1x band and CE keeps declining before
  committing the night.
- **Night watch list**: gate < 0.05 (auto-abort + forensic dump fires),
  pregate > 1 (takeover watch; lambda can be raised), applied p/ans
  divergence, CE trend through the forcing anneal, RSS in `top` (expect
  ~4-5 GB steady).
- **accum 8/16** only if the loss curve looks noise-dominated (wall time
  ~2x/4x per effective step).
- **tau revisit**: if after the night the answer-slot injections dominate
  and prompt-side stays near zero, the 5-8 KV window was the missing
  credit — try batch 4 x accum 8 with `--train-tau 8` (~2x wall time).

### Success criteria

- Held-out EM >= baseline + 5 points on trained tasks, gate demonstrably
  active (nonzero injection on non-trivial steps), free-running stable.
- Digits/trivia unharmed (bench transfer check).
- Gradient diagnostics consistent with the c < 1 contraction model
  (tau-sweep agreement >= 99%).

### Risks and mitigations

- Answer-slot-bias shortcut: early gradient is purely immediate; if that
  alone moves EM, pressure to grow cross-step paths may be weak. Watch the
  cross-step grad share; mitigation: aux local losses (option c).
- Injection takeover (run-5b): pre-gate tax + collapse detector + penalty
  band monitoring.
- Gate collapse at ST entry: scheduled forcing + annealed floor.
- CPU non-determinism: 2-3 runs per claim; checkpoints.
- Training throughput worse than estimated: batch-1 grad-accum fallback;
  shorter prompts.

## Reproduce

```sh
# default overnight recipe (tau=6 exact, batch 4 x accum 4, ~7 h)
cargo run --release -- --cpu --train-enricher --model-dir models/gemma-3-270m-it \
    --train-steps-warmup 150 --train-steps-st 300 \
    --train-out enricher_v2.safetensors

# explicit full form
cargo run --release -- --cpu --train-enricher --model-dir models/gemma-3-270m-it \
    --train-steps-warmup 150 --train-steps-st 300 \
    --train-tau 6 --train-kv-grad --train-batch 4 --train-accum 4 \
    --train-gate-bias 2.0 --train-lr-warmup 50 --train-gate-floor 0.5 \
    --train-inj-l2 0.1 --train-out enricher_v2.safetensors

# tau-sweep diagnostic in isolation (per-process peak RSS is the honest
# number; in-process sweeps ratchet the allocator high-water mark)
cargo run --release -- --cpu --train-enricher --model-dir models/gemma-3-270m-it \
    --train-steps-warmup 0 --train-steps-st 0 --train-skip-eval \
    --train-sweep "6:1" --train-out /tmp/enr_sweep.safetensors

# inference with cascade prefill
cargo run --release -- --cpu --model-dir models/gemma-3-270m-it \
    --enricher enricher_v2.safetensors -p "..." -n 40 --temperature 0
```
