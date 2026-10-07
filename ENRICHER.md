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

**Memory (graph).** ~10-11 MB of retained activations per in-graph step
(18 layers x ~600 KB: q/k/v 4x256 per row, MLP gate/up/act 2048-wide, saved
op inputs). tau=8 -> ~90 MB; tau=16 -> ~170 MB; full BPTT T=80 -> ~0.9 GB
(and backward-prohibitive in time).

**Time.** Forward is memory-bound (1.08 GB weight streaming per
position-step, ~0.13-0.2 s for the whole batch-8 row). Per optimizer step:
~80 cascade forward steps ~ 10-16 s + tau=8 backward ~ 2-4 s + ~5 lm_head
finish calls ~ 0.3-0.6 s => **~15-20 s/step, ~1300-1900 steps/8 h night**.
Warmup 150 + ST 300 ~ 2-3 h. (The M1 two-pass was 6-10 s/step; the ~2x is
the price of training the exact deployment recurrence.)

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
- `induce-d`: the bench induce task + 2-3 distractor statements; known 60%
  baseline band and known feedback sensitivity (Findings 6-7).

4k train + 200 held-out per task; bench induce/digits/trivia remain
transfer measures (no task overlap).

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
4. Data calibration smoke (25/task): baseline EM into the 30-80% band
   under the *new* transition-paired scoring (the old numbers were measured
   under echo pairing; recalibrate).
5. Short CE + ST runs (~30 min): loss curves, gate/inj stats, free-running
   stability.
6. Overnight full run (scale --train-steps-* to the night budget;
   checkpoints every ~5 min).
7. Evaluation: bench `enricher` mode (cascade) vs baseline; held-out EM;
   findings into FEEDBACK.md; commit.

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
