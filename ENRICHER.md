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

## Milestones beyond M1

- **M2**: prefill/prompt enrichment via the *cascading* variant — position
  n+1 processed with the enriched position n already in context (the
  decode-feedback loop run over the prompt with forced tokens; the true
  looped/universal-transformer K-iteration scheme). The parallel two-pass
  prefill feedback exists since Finding 7 and is the ablation baseline.
- **M3**: REINFORCE/GRPO finetune on exact-match reward; task family
  expansion (multi-step state tracking, longer chains).
