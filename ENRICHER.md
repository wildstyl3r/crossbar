# ENRICHER.md — trained "thought pool" enricher: plan, decisions, considerations

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

Findings 6 and 7 (FEEDBACK.md) sharpened this:

- The expected-embedding feedback damages in-context induction
  (60% -> 43%): the induction head needs the *exact* previous-token identity
  as its attention key; the expected embedding smears that key over the
  token's neighborhood (recall survives, precision dies: "68" -> "6").
- Blending with the sampled token (a=0.5) rescues most of the damage
  (-> 56%) and is uniformly non-harmful — the sampled token re-anchors the
  attention key.
- Hand-crafted *prefill* feedback at fixed strength is uniformly
  destructive (0-2% on digit copy at a_p=0.5).

The conclusion: **enrichment strength and content must be learned,
per step** — not hand-set. This document specifies the first trained
mechanism toward that.

## Architecture (M1)

A small trainable cross-attention / attention-pooling module enriches each
token embedding before it enters the first block, drawing on the output
states of already-processed tokens. Backbone (Gemma 3 270M) **frozen**; only
the module trains.

- **State buffer**: last K = 8 entries, each = the previous processing
  step's pre-final-norm last-position output state (the rich "thoughts")
  plus its acted content (the expected embedding used as input). 640-d
  each; trivial memory.
- **Enrichment** (per step, decode):

  ```
  q      = e_t                                  # current token embedding (scaled)
  k_i,v_i = W_k h_i, W_v h_i                    # buffer projections, r = 192
  pooled = sum_i softmax(q^T k_i / sqrt(r)) v_i
  g      = sigmoid(w · [e_t, pooled])           # learned gate (scalar per step)
  input  = e_t * sqrt(h) + g * W_o(pooled)      # W_o zero-initialized
  ```

- **Zero-init injection**: `W_o = 0` at load ⇒ the model starts *bit-exact
  at baseline* and learns where and how much to deviate — the direct
  engineering lesson from Finding 7 (fixed-strength injection destroys
  precision regimes).
- ~0.4M parameters; gate provides the learned, context-dependent blend
  weight (subsumes the hand-crafted alpha).
- With `--enricher <path>` loaded at inference, the enricher *replaces* the
  softmax-blend formula for the fed-back input (the pure formula is kept
  for comparison).
- Inference integration reuses the state-return plumbing: the model again
  returns `(logits, h_t)` from feedback steps, and `generation.rs` pushes
  `(h_t, acted_embedding)` into the buffer.

## Training objective (decided: CE warmup -> ST-Gumbel free-running)

Tasks have **very short verifiable answers** (<= ~4 tokens) by construction,
which keeps the unrolled autoregressive graph tractable on CPU and gives a
low-variance supervised signal — "RL-like with differentiable sampling" in
its most stable form.

1. **CE warmup (teacher-forced)**: vectorized two-pass forward over
   `prompt + known answer`; answer-slot cross-entropy; differentiable and
   stable; initializes the gate. No KV cache in the training graph — the
   in-place candle KvCache is an autograd hazard, sidestepped by a
   cache-free training forward.
   - Pass A (plain embeddings, all positions) captures all-position output
     states; Pass B re-forwards with **only answer-position inputs
     enriched** via causal cross-attention over Pass-A states of positions
     < slot (windowed to K). Prompt positions untouched (decode-only M1).
2. **ST-Gumbel free-running**: generate the short answer autoregressively
   with Gumbel-softmax straight-through sampling (hard one-hot forward /
   soft backward) at answer slots; CE against targets. The whole
   trajectory — including the states the module consumes — is
   differentiable and trained under the module's *own* histories (kills
   exposure bias). Anneal teacher-forced -> ST over steps.
3. **Optional REINFORCE/GRPO finetune** on exact-match reward if CE
   plateaus (group-normalized baseline over G samples per prompt; no value
   network needed). Deferred to M3.

Known M1 approximation: warmup Pass-B consumes Pass-A (unenriched) states
while deployment consumes the enriched forwards' outputs. Second-order at
start (zero-init ⇒ identical), corrected by the ST phase and fully
addressed by the cascading variant in M2.

Optimizer: AdamW, lr 1e-3 -> 1e-4 cosine; batch 8 bucketed by template
length (task templates have fixed length families — near-zero padding);
gradient checkpointing unnecessary at 270M on CPU.

**Gate-collapse guard**: monitor mean gate; if it collapses to ~0 (module
learned to switch itself off), warm up `W_o` without the gate for N steps,
then re-enable the gate.

## Tasks (decided: state tracking + induction-with-distractors)

Synthetic, seeded generators (reusing the bench machinery), held-out split,
baseline EM calibrated into the 30-80% band *before* any long run (the
bench calibration lesson: addition and largest-of-3 were floor-level for
this 270M and were dropped):

- `state-track`: 3 nonsense agents x 2 nonsense objects; chains like
  "The blicket is with zorp. Zorp gives the blicket to kit. The murble is
  with milu. Who has the blicket?" — depth 2-3 with a distractor object
  interleaved; answer = one holder name (1-2 tokens). Probes explicit
  state carrying through the recurrence.
- `induce-d`: the bench induce task (nonsense-word -> number mapping,
  5-shot, 6th queried) + 2-3 distractor statements mixed in; known 60%
  baseline band and known feedback sensitivity (Findings 6-7).

4k train + 200 held-out per task. **Leakage note**: both tasks are new
(no bench-task overlap), so bench induce/digits/trivia remain transfer
measures; digit-copy performance after training on these two tasks is a
clean transfer check.

## Compute reality and execution plan

Environment: 8-core CPU, F32 safetensors backend only (GGUF cannot train).
Measured decode ~110-180 ms/token; training step (batch 8, two vectorized
passes through 18 frozen layers) estimated 6-10 s/step ⇒ a first
meaningful run is a **multi-hour overnight job** (user-approved:
unattended runs with ~30-min checkpoints are acceptable).

Execution sequence (each step gates the next):

1. ~~Finish bench round 2; Finding 7 into FEEDBACK.md; commit everything.~~
2. `src/enrich.rs` (module + VarMap save/load) + model integration +
   `--enricher` inference flag; regression battery (committed runs
   bit-identical with no enricher; greedy T->0 equivalence; probe).
3. **Gradient sanity test** (retire the one real technical risk first):
   toy trainable module inside the frozen forward — loss must decrease and
   grads must flow *only* to the module's VarMap params.
4. Data calibration smoke (25/task): baseline EM into the 30-80% band;
   adjust depth/distractor counts.
5. Short CE run (~15 min): loss curve, held-out EM movement, gate stats.
6. Short ST-Gumbel run (~30 min): EM stability, buffer/gate sanity.
7. Overnight full run (~2-4 h, batch 8, checkpoints every ~30 min via
   VarMap -> safetensors export).
8. Evaluation: `enricher` bench mode vs baseline / blend-d0.5 on the bench
   suite + held-out EM; Finding 8 into FEEDBACK.md; commit.

### Success criteria

- Held-out EM >= baseline + 5 points on trained tasks, with the gate
  demonstrably active (nonzero injection on non-trivial steps).
- Digits/trivia unharmed (bench transfer check).
- Secondary: does the learned pooler beat blend-d0.5's 56% on induce at
  the same information budget?

### Risks and mitigations

- candle autograd through frozen ops — gated by the step-3 sanity test;
  if broken, fall back to manual backward for the small module.
- In-place KvCache inside autograd — eliminated by design (cache-free
  training forward).
- Task calibration off — smoke run before any long training.
- CPU throughput worse than estimated — 30-min checkpoints make partial
  runs useful; batch-1 grad-accum fallback.

## Interim findings from the first training runs (runs 1-4)

Implementation progress so far: enricher module (`src/enrich.rs`, w_q/w_k/
w_v/w_o/gate, zero-init W_o, VarMap + safetensors save/load), cache-free
pad-aware training forward (`forward_inputs_train`, prompt-length bucketed
batches so RoPE positions match inference exactly), CE warmup + guided
ST-Gumbel phases, dual evaluation (teacher-forced argmax EM + free-running
greedy EM), 30-min checkpoints.

What the runs taught us, in order of importance:

1. **The capitalization bug had two layers, and only one was obvious.**
   The model naturally emits capitalized names ("Milu", "Zorp"); training
   on lowercase targets ("milu") made the CE objective near-impossible
   (~17-25 nats/token), and the gate learned the escape hatch: it switched
   *itself off* (gate -> 0), which zeroes the W_o gradient and freezes the
   module while the loss stays high. Fixes: (a) capitalized targets, and
   (b) scoring was *also* case-sensitive (`w == answer` against lowercased
   text) — every state-track example scored false regardless of the
   model's output, silently corrupting all free-running EM numbers
   reported earlier. Both fixed; scoring is now case-insensitive.

2. **Gate dynamics need protection on both ends.** Run 1 (lowercase
   targets): gate collapsed to 0 (escape hatch). Run 2 (capitalized):
   converged beautifully (loss 17 -> 0.7-2.3, gate pinned at 1.0) but its
   free-running eval collapsed into loop attractors ("KitMilMilMil…") —
   the exposure mismatch between teacher-forced histories and free-running
   generation, plus a free-running eval that had to restart from an
   essentially untrained cold buffer. Run 3 (same seeds as run 2!) *still
   collapsed* (gate -> 0, loss climbing) — see finding 3. Run 4 adds:
   LR warmup over the first 50 steps (first-step CE ~17-25 produces huge
   gradients), gate bias initialized at +2.0 (sigmoid ~ 0.88: the gate
   starts OPEN and must learn to close, instead of starting closed at 0.5
   and having to fight its way open), and a warmup-only gate floor of 0.5
   (gate = floor + (1-floor)·sigma(...) so an early mis-step cannot shut
    the module off and zero the W_o gradient). Run 4 trajectory: loss
    8.7 (step 10) -> 1.5-3.5 oscillating, gate alive at 0.5-0.6 — no
    collapse through warmup. **Run 4 terminal: the ST phase still collapsed
    (gate -> 0 by ST step ~20, loss ~14-15 thereafter); the warmup gate
    floor does not carry over into free-running, where unanchored
    trajectories produce large, useless gradients.**

3. **candle CPU matmuls are not run-to-run deterministic.** Two runs with
   identical seeds, identical code, identical data: step-0 loss identical
   (17.2452) but divergence by step 10 and completely different
   trajectories (run 2 converged; run 3 gate-collapsed). Cause: thread
   scheduling in parallel CPU reductions changes float summation order;
   AdamW on a sharp loss surface amplifies tiny differences. Implication:
   **fixed seeds do not guarantee reproducible training trajectories on
   CPU**; checkpoints (kept) are the safety net, and run-to-run variance
   must be treated as an experimental variable (report across 2-3 runs
   when claiming effects).

4. **Exposure bias is the dominant M1 gap** (from run 2: healthy
   teacher-forced loss ~1.6, free-running loop attractors). The guided
   ST-Gumbel phase exists precisely for this; its effectiveness is what
   run 4's final dual evaluation will show.

## Runs 5-6 (2026-10-06): the M1 terminal verdict

Short-schedule redo (75 warmup + 150 ST, justified by run 2's own trace),
with the run-4 extras made configurable (`--train-gate-bias`,
`--train-lr-warmup`, `--train-gate-floor`, `--train-gt-start/end`,
`--train-seed`; run-2-faithful defaults). Also added: a gate-collapse
detector (auto-abort after gate < 0.05 for 60 steps, forensic checkpoint),
and the **first-token-guided free-running eval** (target's first token
teacher-forced, then free-run) — which separates first-step bootstrap
failures from decoding destabilization.

- **Run 5a** (run-2-faithful: no ramp/floor/bias): warmup failed (loss
  6-17 oscillating at baseline level, gate 0.3-0.5, never opened) —
  killed manually. Note: step-0 loss identical across runs (17.2452) is
  uninformative — zero-init W_o makes step 0 baseline regardless of the
  module's (unseeded, thread_rng) weight draw; each process rolls fresh
  module weights.
- **Run 5b** (hybrid: lr ramp 50 + bias 0 + no floor): trained healthy
  (warmup -> 3.2, ST 2.1 -> bump 5.9 -> **1.47**, gate 1.000 throughout).
  Eval matrix: TF EM 27.6 -> 25.9 (flat), FR EM 33.0 -> **2.5** (loop
  attractors), guided FR EM **88.0 -> 13.0** — the module actively
  destabilizes decoding even with a correct first token. Same failure as
  run 2, now reproduced from a clean training run with a kept checkpoint.
- **Probe** (`--train-probe`, run-5b checkpoint): the Gumbel mixtures were
  barely smeared (c_in 0.95-0.97 — no denoising opportunity); the module's
  output is near-orthogonal to the clean embedding (c_st 0.21-0.22) and to
  the input at deployment (c_dep 0.01-0.03); injection magnitude
  **~131x the input norm**; ST inputs double-scaled by sqrt(h)
  (`expected_from_probs` pre-scales, `enrich()` scales again; scale
  measured 22-24 vs sqrt(640)=25.3); gate pinned at 1.0 (saturated by the
  25x-scale inputs). **Diagnosis: the module learned to REPLACE the token
  identity with W_o(pooled(states)) — the frozen model's own contextual
  predictions — which is CE-optimal at training and precision-destroying
  at deployment (the Finding-6 mechanism). Nothing in the objective
  bounded the deviation.**
- **Run 6a** (hard-forward ST + single-scaling + relative-L2 injection
  penalty λ=0.1, no floor): the gate learned a new escape hatch — closing
  it dodges the *applied* tax (penalty was on `gate * injected`) — and
  collapsed at ST entry; auto-abort worked as designed.
- **Run 6b** (+ gate floor 0.3 through both phases): trained through
  (ST -> ~0.9-2.7), but equilibrated at the **identity** (gate rides the
  annealed floor; applied injection ~0.1x). Eval: TF 27.6 = 27.6 (flat),
  FR **31.5** (not destroyed), guided FR **87.0** (not destroyed). Probe:
  c_dep 1.000, inj 0.000 — bit-exact identity at inference.

**M1 sweep verdict**: at 131x and 0.1x injection, answer-slot enrichment
from last-8 Pass-A states produced **zero** teacher-forced gain — the M1
information source carries little extractable signal for these tasks.
Two incentive lessons carried forward: (a) tax the **pre-gate** injection
(the applied tax lets the gate dodge), (b) a magnitude penalty without a
floor is escapable, a floor without a tax is a blindfold. M2
(cascade-trained prompt-position enrichment — where the hand-set cascade
already demonstrated +18) is the next design.

### Teacher-forcing check (run 4's terminal evaluation; what to look for)

Run 4 ends with a dual evaluation on 200 held-out examples:

- **teacher-forced EM, baseline (Pass-A logits) vs trained (Pass-B logits
  with enriched answer slots)**: isolates the module's one-shot effect on
  answer-slot predictions from generation dynamics. Baseline TF EM was
  measured at 27.6% (note: with the fixed scoring this number needs
  re-measurement; it was corrupted by the case bug for state-track).
  Success criterion: trained TF EM >= baseline TF EM + 5 points.
- **free-running EM, fresh-enricher (= baseline identity) vs trained
  enricher**: the deployment-realistic number (states buffer + enrichment
  active during generation). Baseline FR EM ~17% (same caveat).
- Interpretation matrix:
  - TF up, FR up: module learns useful enrichment; proceed to cascade +
    bench transfer.
  - TF up, FR flat/down: enrichment helps one-step predictions but the
    buffer/state distribution differs under free running — the M1
    approximation (Pass-A states at warmup vs enriched deploy states) and
    exposure bias dominate; strengthen the ST phase or move to cascade
    training (below).
  - TF flat: the module found nothing beyond copying baseline logits;
    check gate stats and window relevance before more compute.

## Cascade decoding of the input (the user's original intent)

The initial idea — position n+1 processed with the *enriched* position n
already in its context, so each position's emergent distribution reflects
already-enriched thinking — is the decode-feedback loop run over the prompt
with forced tokens. Two realization schedules:

### Schedule A: pure sequential cascade (per position)

One forward per prompt position, each position's input enriched from the
processed states of all previous positions. Inference: trivial with KV
caches (~65 forwards/prompt ≈ 7 s — fine). Training: 65 sequential
forwards per example per step — **not viable** for gradient training on
CPU. Rejected for training; usable for inference-only probes.

### Schedule B: chunked/iterated cascade (recommended)

Observe that the existing `forward_prompt_blend` already implements
**iteration 1** of the cascade: the observation pass (iteration 0) produces
per-position logits; the injection pass (iteration 1) forwards the blended
inputs and builds the cache. A true cascade is simply:

```
states/logits_0 = forward(E_prompt)                      # iteration 0
for k in 1..=K:
    inputs_k = (1-a) * E_prompt + a * expected(logits_{k-1})
    logits_k = forward_all_positions(inputs_k)           # fresh caches
# decode continues from iteration K's cache + last-position logits
```

- Cost: (K+1) full-sequence forwards (each ~1 prefill) + per-iteration
  full-position logits (finish() over all positions; 64x262k F32 ≈ 67 MB
  transient — fine). K=3 ⇒ ~4x prefill.
- This is the **looped/universal-transformer schedule** with our blend
  formula as the refinement operator: each iteration re-thinks every
  prompt position with the *refined* context of the previous iteration in
  its KV cache; the emergent distributions compound.
- Known gap vs Schedule A: within one iteration, refinement is
  chunk-parallel (all positions refined against the *same* previous
  iteration), so the within-iteration sequentiality of Schedule A is
  lost; K iterations approximate it (K=1 one-shot < K=2 < ... <=
  sequential).
- Implementation: moderate (~1-1.5 h). Restructure `forward_prompt_blend`
  into the iteration loop above (`--prefill-feedback-iters K`, default 1
  = current behavior; per-iteration full-position logits via `finish`
  without the last-position narrow). Regression: K=1 must reproduce
  current `prefill-p0.5` behavior exactly.
- Training-time recipe (deferred): iterate the two-pass scheme K times
  with **stop-grad on previous-iteration states/logits** (treat them as
  constants; gradients flow through the current forward into the
  enricher only). Cost K x 2 forwards per step (K=2 ≈ 12-20 s/step —
  feasible). This directly trains the module under refined histories —
  the proper fix for the M1 warmup approximation.

### Minimalist cascade experiment (run 2026-10-06; inference-only, no training)

Implemented as `--prefill-feedback-iters K` (Schedule B; iteration 0 = the
original observation pass, bit-exact for K=1 — regression battery passed:
CLI text and bench CSV identical to the pre-change binary for
baseline/blend-d0.5/prefill-p0.5/both-0.5). Probes as designed, plus two
extra bench columns that the P0 dynamics made necessary (cascade-k3,
cascade-k2-a25).

**P0 dynamics** (long France paragraph ending mid-sentence, greedy; step-1
raw_H vs baseline 1.10, "The answer is **Paris."):

| config | step-1 raw_H | continuation | verdict |
|---|---|---|---|
| K=1, a_p=0.25 | 0.21 | "The capital of France…" | healthy |
| K=2, a_p=0.25 | 0.35 | "The capital of France…" | healthy |
| K=3, a_p=0.25 | 0.71 | "The answer is **Paris." (≈ baseline) | healthy |
| K=1, a_p=0.50 | 5.50 | "The country has a…" | destroyed (Finding 7) |
| K=2, a_p=0.50 | 4.19 | "The country has a…" | destroyed (worse than K=1) |
| K=3, a_p=0.50 | 3.21 | "The capital of France is Paris." | repaired (entropy elevated) |

Repair is **non-monotone in K at fixed strength**: at a_p=0.5 the
compounding lookahead first deepens the destruction, then iteration turns it
around (K=3). Per-iteration prefill entropies (diagnostics): a_p=0.5 →
1.36 → 2.91 → 2.67 (does not sharpen); a_p=0.25 → 1.36 → 1.56 → 1.55.

**Bench triage** (greedy, paired; triage N=25, then the winning config
confirmed):

| task | baseline | blend-d0.5 | prefill-p0.5 | cascade-k2 (a.5) | cascade-k3 (a.5) | cascade-k2-a25 |
|---|---|---|---|---|---|---|
| digits-5 | 100% | 100% | 0% | 8% | 4% | 88% |
| digits-10 | 96% | 96% | 0% | 52% | 0% | 84% |
| induce | 64% | 60% | 4% | 12% | 4% | **80%** |

(triage N=25, McNemar vs baseline in logs/cascade_bench25*.log)

**Confirm run** (cascade-k2-a25 = K=2 iterations at a_p=0.25):

| task | baseline | cascade-k2-a25 | p (McNemar) |
|---|---|---|---|
| induce (N=100) | 60% | **78%** | **0.006** |
| trivia (N=40) | 72% | 72% | 1.000 |
| digits-5 (N=50) | 100% | 94% | 0.250 |
| digits-10 (N=50) | 96% | 88% | 0.219 |

→ **Finding 8** (FEEDBACK.md): the refinement *schedule* itself carries
value — at gentle strength the iterated cascade improves induction by +18
points over baseline (and over blend-d0.5's 56%) with no collateral damage,
before any training.

**Decision (branch taken): cascade-helps ⇒ cascade-trained ST overnight run
(K=2, stop-grad recipe) is prioritized over plain M1 ST** — after the run-4
ST gate-collapse fix (below), since no trained enricher exists yet (the
run-4 checkpoint is post-collapse ≈ identity). The a_p≥0.5 / K=3
destruction is the strongest argument for the learned gate: strength must
be context-dependent, not a fixed constant.

Caveats: single-run (CPU run-to-run variance, Finding 3 — but greedy
inference is deterministic in practice); induce mean generated length rose
(6.5 vs 3.8 tokens — some answers run past the number before EOS), worth
watching in the trained variant.

## Strategy outline (updated)

1. ~~Bench round 2 + Finding 7 + commit.~~ (done, edaf828)
2. ~~Enricher module + integration + gradient sanity.~~ (done; runs 1-4)
3. ~~Run 4~~ terminal state: warmup healthy, ST collapsed (gate -> 0 by
   step ~20; loss ~15); checkpoint post-collapse (~= identity).
4. ~~Minimalist cascade experiment (`--prefill-feedback-iters`)~~ (done
   2026-10-06; **cascade-helps branch**: Finding 8, induce +18 p=0.006 at
   K=2/a_p=0.25).
5. ~~M1 short redo (runs 5-6)~~ (done 2026-10-06; **M1-negative verdict**:
   the module either replaces the token identity (131x injection, deployed
   decoding destroyed) or equilibrates at the identity; TF EM flat at both
   operating points — see "Runs 5-6" above).
6. **M2: cascade-trained prompt-position enrichment** — the module learns
   the residual refinement on top of the hand-set cascade operator
   (K=2, a_p=0.25 floor at 78% induce); inherits the hard-forward sampled
   inputs, the pre-gate injection tax, floors, and the collapse detector.
7. Bench the trained M2 module vs baseline / hand-set cascade
   (transfer check: digits/trivia unharmed) -> FEEDBACK.md -> commit.
8. M3: REINFORCE/GRPO finetune; task family expansion.
