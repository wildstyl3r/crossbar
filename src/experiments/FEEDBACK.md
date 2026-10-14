# Softmax feedback mode: experiments and findings

This document records the design rationale and experimental findings for the
softmax feedback mode (`--softmax-feedback`), in which the autoregressive
decode loop feeds the previous step's **expected token embedding** back into
the first transformer block instead of embedding the sampled token.

All experiments below used the `gemma-3-270m-it` safetensors backend on CPU
(F32), vocab 262,144, hidden size 640 (embedding scale sqrt(640) ≈ 25.3),
tied LM head, unless stated otherwise. Default sampling: temperature 1.0,
top-k 64, top-p 0.95, repeat penalty 1.1 over last 64.

## Mechanism

Normal decode step: sample token `a_t`, feed `E[a_t] * sqrt(h)` into block 0.

Feedback step: feed

```
x_t = softmax(logits_{t-1} / T) @ E * sqrt(h)
```

— the expected embedding under the model's own next-token distribution. This
keeps the input **within the convex hull of the token embeddings** (on-manifold),
unlike a raw hidden state, which is off-manifold and noisy (an earlier
prototype of hidden-state feedback produced fluent-looking but far more
degenerate text, e.g. Japanese/French/emoji drift; it was removed in favor of
this variant).

The textual path (sampling, printing, repeat penalty, EOS stop) is completely
untouched; the feedback only changes what the *model* sees.

Properties:

- `T -> 0`: one-hot feedback, i.e. exactly the argmax token's embedding.
  Verified experimentally: greedy sampling + T = 1e-6 reproduces normal greedy
  generation **bit-for-bit** (a one-hot matmul with the tied embedding table
  equals the embedding lookup).
- `T -> inf`: uniform, i.e. the mean embedding — semantically empty input.
- `T = 1`: the model's own distribution, unmodified.

## Per-step metrics (diagnostics)

`--softmax-feedback-diagnostics` prints one CSV row per step to **stderr**
(stdout stays clean text):

| column      | meaning                                                                 |
|-------------|-------------------------------------------------------------------------|
| `step`      | decode step index (0 = prefill)                                         |
| `T`         | feedback temperature actually used (NaN at prefill)                     |
| `H_fb`      | entropy of the feedback distribution, nats                              |
| `support`   | effective support `exp(H_fb)`                                           |
| `norm_x`    | L2 norm of the fed-back embedding (real tokens ≈ 25 here)               |
| `cos_prev`  | cosine between consecutive fed-back inputs (NaN on first feedback step) |
| `raw_H`     | entropy of the model's raw distribution at T=1                          |
| `surprisal` | −log p_raw(sampled token)                                               |
| `distinct2` | distinct bigrams / bigrams over the last 32 generated tokens            |

`H_fb` and `norm_x` measure **input smearing** (peaked distribution → one
token-like embedding; flat distribution → small-norm centroid). `cos_prev`
measures **stagnation** (consecutive inputs nearly parallel ⇒ fixed point /
attractor). `raw_H`, `surprisal`, `distinct2` are output-side ground truth.

## Finding 1: fixed-T degradation spectrum

Prompt "Once upon a time", seed 7, 40 tokens:

| T    | behavior                                                                    |
|------|-----------------------------------------------------------------------------|
| 0.5  | coherent narrative                                                          |
| 1.0  | essentially normal generation (the distribution's mode dominates)           |
| 2.0  | coherent for ~20 tokens, then locks into a word-list attractor              |
|      | ("…acacia acacia acacia chaise bookcasecento acacia koffcento skyscraper…") |
| 5.0  | multilingual word salad, but still word-like                                |
| 20.0 | near-mean-embedding collapse; output is blank lines                         |

## Finding 2: the degradation signature is measurable in advance

Diagnostics on the T=2.0 run — the transition into the attractor at step ~22
is unambiguous, and all three hypothesized signals fire together:

- `H_fb` runs away: ~0.1–6 nats during the coherent phase → **11.4 nats**
  (support ≈ 90k of 262k tokens) inside the attractor.
- `cos_prev` locks: 0.2–0.7 → **0.9998** (the smeared centroid stops moving).
- `raw_H` jumps 0–3 → **~7 nats**: fed a broad centroid, the model predicts
  nearly everything at once.
- `norm_x` sags 25 → 11 (real-token scale → centroid scale).
- `distinct2` drops 1.000 → ~0.93 (repeated bigrams appear).

So entropy of the feedback distribution and consecutive-input cosine are
sufficient early-warning signals; no output-side metric is needed for
*prediction* (they serve as ground truth for validation).

## Finding 3: the entropy servo works, in both directions

Because H(softmax(logits / T)) is monotone non-decreasing in T
(dH/dT = Var(logits)/T³), the temperature can be *solved* per step:

```
T_t = { T : H(softmax(logits_t / T)) = H* }   (bisection, clamp [1e-3, 20])
```

`--softmax-feedback-mode entropy-servo --softmax-feedback-target-entropy 2.5`
on the same acacia config:

- `H_fb` locked to **2.45–2.57 on every step** (target 2.5); no runaway,
  `cos_prev` max 0.91, `distinct2` = 1.000.
- The solved T adapts **bidirectionally**: 0.58–2.80. When the model is
  overconfident (low raw_H) the servo raises T to keep the input diverse;
  when raw entropy spikes it cuts T hard (e.g. step 37: raw_H 4.6 → T 0.62).
- Text becomes fluent narrative: *"Once upon a time, there lived a small girl
  named Flicker. She was bright and with eyes green, always hair the like
  shade of dandelion…"* — the "colorful but tamed" regime, vs the acacia
  attractor at fixed T=2.
- Overhead: ~35 ms/step on CPU (≈14 softmax passes over the 262k logits).

## Finding 4: the threshold heuristic also works, but is choppier

`--softmax-feedback-mode heuristic` (band ±1.0 nat around H*, decay γ=0.8,
stagnation trigger cos > 0.999) also held the line on the acacia config, but
T oscillates between band edges (1.28–3.13) instead of settling. Notably the
stagnation (cosine) trigger never fired — the entropy band did all the work;
the consecutive-cosine signal was redundant *here* (it may still matter for
low-entropy loops). Removed from the CLI after Findings 3/5: the one-sided
servo dominates it (settles instead of oscillating, and tames without
blurring confident regimes).

## Finding 5: forcing diversity hurts confident regimes — hence the one-sided servo

Greedy sampling (temperature 0, repeat penalty 1.0), prompt "Write a list of
ten things to pack for a camping trip":

- **Baseline**: clean, complete 10-item list, 200 tokens.
- **entropy-servo (H\*=2.5)**: structure kept, content degraded ("A essential
  for staying dry", "First-P"). In a confident regime (raw_H ≈ 0) the servo
  must *inflate* T well above 1 to reach H\*, blurring inputs the model
  already meant precisely. Garbage-in at the semantic margin.
- **One-sided servo** (`--softmax-feedback-mode one-sided-servo`): solves T as
  above but **caps it at 1.0** — the servo only ever *tames* (T < 1 when the
  raw distribution is flatter than H\*) and otherwise falls back to the
  model's own distribution (T = 1) instead of blurring. Results:
  - acacia config: still fully tamed (T mostly 1.0, dipping to 0.948 exactly
    when raw_H exceeded H\*; coherent text, no attractor);
  - greedy list: much better than the two-sided servo (structure and item
    identities preserved, e.g. "First-Aid Kit", "Cooking Stove & Fuel"), EOS
    at 178 tokens;
  - but still slightly below baseline: even at T = 1 the full distribution's
    tail mass blurs the input relative to the discrete sampled-token
    embedding ("Water::", "Cooking Ut" slips remain).

## Conclusions

- The expected-embedding feedback places the recurrent state on the token
  manifold; its failure mode is **smear collapse** (flat distribution →
  centroid input → attractor), reliably predicted by `H_fb` + `cos_prev`.
- The **one-sided entropy servo** is the recommended configuration: it
  eliminates the failure mode while leaving confident generation at (nearly)
  the model's own distribution. Treat H\* as "maximum input diversity in
  nats": 2.5 ≈ a 12-token semantic neighborhood.
- The servo is a *diversity controller*, not a quality enhancer: any input
  blur above the model's own distribution costs quality in confident regimes.
- Possible follow-ups: truncate/renormalize the feedback distribution
  (top-k/top-p mass) to kill tail blur at T = 1; per-layer or per-position
  temperatures. (The third follow-up — blending the fed-back input with the
  sampled token's embedding — was implemented and measured: Finding 7.)

## Finding 6: capability impact measured with `--bench`

Because the feedback input is only constructed during multi-step generation,
any teacher-forced (log-likelihood) benchmark is blind to it by construction.
`--bench` therefore runs generate-then-score tasks in-process, greedy
decoding, repeat penalty 1.0, identical paired prompts across modes
(deterministic seeds), 100 examples per task (40 for trivia). Candidate tasks
were calibrated first: 2-digit and even single-digit addition, and
largest-of-3 comparison land at 4–24% for this 270M model (floor — the model
echoes operands or the last list item), so the final suite keeps only tasks
where baseline EM sits in the 60–100% band.

Suite: `digits-5/10/20` (copy a random digit string; per-step state
fidelity), `induce` (5-shot nonsense-word→number mapping, 6th queried;
in-context learning), `trivia` (40 curated factual Q&A; semantic recall).
Modes: baseline, fixed T=1.0 (isolates expected-embedding-vs-token from servo
effects), one-sided servo H\*=1.5, one-sided servo H\*=2.5. Significance via
exact McNemar on paired outcomes.

| task        | baseline | fixed-T1.0 | one-sided-H1.5 | one-sided-H2.5 |
|-------------|----------|------------|----------------|----------------|
| digits-5    | 100%     | 100%       | 100%           | 100%           |
| digits-10   | 96%      | 96%        | 96%            | 96%            |
| digits-20   | 100%     | 100%       | 100%           | 100%           |
| induce      | **60%**  | 43% (p<0.001) | 47% (p=0.001) | 44% (p<0.001) |
| trivia      | 72%      | 72%        | 72%            | 72%            |

Mean generated lengths are identical across modes (no early-EOS artifact).

Interpretation:

- **Copy/recall and semantic trivia are completely robust** to the feedback
  across all modes — including 20-digit copies (~40 decode steps). Both are
  max-confidence regimes (raw_H ≈ 0.06): the feedback distribution is
  essentially one-hot, so the expected embedding ≈ the argmax embedding.
- **In-context induction is significantly damaged**: 60% → 43–47%
  (13–17 points, p < 0.001). The failure mode is precise and mechanical: in
  the servo CSV, 17 examples flipped right→wrong vs 1 wrong→right, and the
  typical wrong answer is the correct mapping **truncated to its first
  digit** ("68"→"6", "14"→"1", "69"→"6"). The induction head needs the exact
  previous-position token identity to look up the in-context mapping; feeding
  the expected embedding smears that key over the distribution's
  neighborhood, and fine-grained precision (the second digit) is lost.
- The one-sided servo does not rescue induction (44–47%): since induce raw_H
  ≈ 0.5–0.6 < H\*, it runs at T = 1 — i.e. the damage comes from the tail
  mass of the expected embedding at T = 1 itself, not from servo-inflated
  temperatures. (Fixed-T1.0 vs baseline isolates exactly this effect.)
  H\*=1.5 scores marginally best (47%) — the servo's occasional T<1 taming
  when raw_H > 1.5 helps slightly.
- Practical takeaway: expected-embedding feedback is safe for confident,
  precision-doesn't-matter regimes (fluency, style, topic continuation) and
  costs measurable ICL precision. If ICL preservation matters, the next
  lever is **truncating the feedback distribution** (e.g. renormalize the
  top-k mass before the embedding mixture) so the input stays near the
  sampled token while still bounding worst-case smear.

Reproduce: `cargo run --release -- --cpu --bench --bench-n 100` (per-example
rows land in `bench_results.csv`; calibration used `--bench-n 25`).

## Finding 7: blending rescues most of the induction damage; parallel prefill feedback is destructive

Two extensions of the feedback path were bench-tested (both are CLI knobs of
the same mechanism): **decode blending** mixes the expected embedding with
the sampled token's embedding — re-anchoring the exact attention key the
induction head needs while keeping the neighborhood mass:

```
input_t = (1 - a) * E[a_t] * sqrt(h)  +  a * (softmax(logits_{t-1} / T) @ E) * sqrt(h)
```

and **prefill feedback** applies the same idea to prompt processing: an
observation pass over the prompt builds per-position expected embeddings,
blended into the injection pass that builds the KV cache the decode loop
continues from (`a_p`, two-pass, chunked; `a_p = 0` default keeps prefill
untouched). Bench round 2 modes: `blend-d0.5` (decode blend a=0.5, T=1),
`prefill-p0.5` (prefill blend a_p=0.5, normal decode), `both-0.5` (both).
Same protocol as Finding 6, greedy, paired, N=100/40:

| task      | baseline | blend-d0.5 | prefill-p0.5 | both-0.5 |
|-----------|----------|------------|--------------|----------|
| digits-5  | 100%     | 100%       | 1%           | 2%       |
| digits-10 | 96%      | 96%        | 0%           | 0%       |
| digits-20 | 100%     | 100%       | 0%           | 0%       |
| induce    | 60%      | 56% (p=0.125) | 3%        | 4%       |
| trivia    | 72%      | 78% (p=0.500) | 45% (p=0.001) | 50% (p=0.004) |

- **Decode blending works as designed.** a=0.5 recovers roughly three
  quarters of the pure-feedback induction damage (43% -> 56%, baseline 60;
  the remaining gap is not significant at N=100, p=0.125) while leaving
  digits exactly at baseline and nudging trivia up (72 -> 78, ns). The
  sampled-token re-anchor preserves the exact attention key the induction
  head needs, as predicted.
- **Parallel prefill feedback at a_p=0.5 is uniformly destructive** — every
  task collapses (digits 0-2%, induce 3-4%, trivia 45-50%), and generated
  lengths balloon to the token caps (21/32/52 tokens = degeneration). Note
  the prefill blend is a *lookahead* injection (each position receives the
  expected next-token embedding derived from that position's own logits),
  which at 50% strength overwhelms the early-layer processing of the whole
  context. Even semantic trivia (weight-stored knowledge) loses a quarter of
  its accuracy.
- Conclusion: the enrichment *strength* (and where it is applied) cannot be
  a fixed constant — it must be learned, per step. This directly motivates
  the trained cross-attention enricher with a learned gate
  (see ENRICHER.md), which starts at exact baseline (zero-init injection)
  and learns when and how much to enrich.

## Finding 8: iterated prefill cascade at gentle strength improves induction (+18, p=0.006) without collateral damage

Implementation: `--prefill-feedback-iters K`. Iteration 0 is the existing
observation pass; iterations k = 1..K re-forward
`(1 - a_p) * E_prompt + a_p * expected(logits_{k-1})` over all positions
with fresh caches; decode continues from iteration K's cache. Cost:
(K+1) full prefills. K=1 reproduces the old `prefill-p0.5` bit-exactly
(regression battery passed).

Dynamics probe (long France paragraph, greedy; step-1 raw_H, baseline 1.10):

| config | raw_H | continuation |
|---|---|---|
| K=1, a_p=0.25 | 0.21 | "The capital of France…" (healthy) |
| K=2, a_p=0.25 | 0.35 | healthy |
| K=3, a_p=0.25 | 0.71 | ≈ baseline text |
| K=1, a_p=0.50 | 5.50 | destroyed (Finding 7) |
| K=2, a_p=0.50 | 4.19 | destroyed (worse than K=1) |
| K=3, a_p=0.50 | 3.21 | repaired: "The capital of France is Paris." |

Bench (greedy, paired, tasks digits-5/10 + induce; triage N=25, confirm
N=100/50 for the winning config):

| task | baseline | prefill-p0.5 | cascade-k2 a.5 | cascade-k3 a.5 | cascade-k2 a.25 |
|------|----------|--------------|----------------|----------------|-----------------|
| digits-5  | 100% | 0% | 8%  | 4% | 88% → 94% (conf., p=0.250) |
| digits-10 | 96%  | 0% | 52% | 0% | 84% → 88% (conf., p=0.219) |
| induce    | 64% (60 conf.) | 4% | 12% | 4% | 80% → **78% (conf., p=0.006)** |
| trivia (conf.) | 72% | — | — | — | 72% (p=1.000) |

Interpretation:

- **Strength dominates depth.** At a_p=0.5 the lookahead injection
  compounds: K=2 is worse than K=1 and K=3 only turns the *dynamics*
  around (not the bench). At a_p=0.25 the cascade is uniformly safe at all
  K probed.
- **The iterated cascade at gentle strength is the first prompt-side
  mechanism that *improves* ICL**: induce 78% vs 60% baseline
  (+18, p=0.006, N=100), also above blend-d0.5 (56%). Each prompt position
  receives 25% of the previous iteration's expected continuation of its
  own output while the exact token keys stay anchored at 75% — the
  induction head keeps its precise keys and gains anticipatory context.
- digits/trivia essentially unharmed (digits-5 94% / digits-10 88% vs
  100/96, both ns at N=50; trivia identical). Watch item: induce mean
  generated length rose (6.5 vs 3.8 tokens — some answers run past the
  number before EOS).
- This is an inference-only effect (no training) measured with the
  hand-set blend operator; the learned per-step gate (ENRICHER.md) is
  expected to own the strength schedule — the a_p=0.5 destruction shows
  fixed strength cannot be rescued by depth alone.

Reproduce:

```sh
# dynamics probe (per-step CSV on stderr)
cargo run --release -- --cpu -p "<long France paragraph>" -n 40 --seed 7 \
    --temperature 0 --softmax-feedback-diagnostics \
    --prefill-feedback-blend 0.25 --prefill-feedback-iters 2

# bench (cascade modes; filters select the subset)
cargo run --release -- --cpu --bench --bench-n 100 --bench-tasks induce,trivia \
    --bench-modes baseline,cascade-k2-a25
```

## Bench validity note (2026-10-08): `induce` queries are always the last-stated pair

Auditing the bench for train/eval leakage surfaced a task-design issue that
reinterprets Finding 6's induce column:

- **The original bench `induce` never shuffles its statements** (bench.rs
  `build_examples`): 5 pairs are drawn in order and `query = chosen[4]`,
  so the queried pair is always the LAST stated one. Verified empirically:
  **400/400 induce rows in `bench_results.csv`** have the queried word as
  the final "X is N" statement before the question.
- **Consequence**: the task is solvable by "echo the last stated number"
  with zero word->number binding. The Finding 6 damage signature
  ("68" -> "6", "14" -> "1") is a generation truncation of the *correct*
  value — consistent with last-number copying as much as with
  induction-head lookup. The 60% -> 43-47% feedback damage should be read
  partly as degraded positional-copy precision, not purely as in-context
  induction damage.
- **In-family, not transfer**: bench `induce` also shares the statement
  and question format, the number range, and 15/16 vocabulary words with
  the enricher training task induce-d, and training prompts cover every
  (word -> number) pair ~8x (12k pair-instances over 15 x 99 = 1,485
  combos). This is unexploitable — the mapping is random per example, so
  no consistent association exists to memorize, and the answer always
  derives from the prompt's own context — but it means bench induce
  measures the same skill family the enricher trains on, not an
  independent capability. digits/trivia are unaffected (no training
  counterpart).

Remediation (in bench.rs, old task kept for comparability with Findings
6-8):

- `induce-sh`: same 5-pair task with statements shuffled (mirrors the
  induce-d generator) — the queried pair lands at a random position and
  the last-number shortcut fails.
- `induce-early`: the queried pair is drawn from positions 0..4 (never
  last-drawn) with shuffled statements — the answer requires binding the
  query word to its stated number anywhere in context; last-number
  copying is correct only by chance (~1/5).

For enricher verdicts, cite `induce-sh` / `induce-early`; treat the
original `induce` as a copy-precision + in-family diagnostic.

Reproduce: `cargo run --release -- --cpu --bench --bench-n 100
--bench-tasks induce,induce-sh,induce-early --bench-modes baseline`

## Example commands

```sh
# fixed temperature
cargo run --release -- --cpu -p "Once upon a time" -n 40 --seed 7 \
    --softmax-feedback --softmax-feedback-temperature 2.0

# entropy servo (recommended), with per-step diagnostics on stderr
cargo run --release -- --cpu -p "Once upon a time" -n 40 --seed 7 \
    --softmax-feedback --softmax-feedback-mode entropy-servo \
    --softmax-feedback-target-entropy 2.5 --softmax-feedback-diagnostics

# one-sided servo (tames only; closest to normal generation when confident)
cargo run --release -- --cpu -p "Once upon a time" -n 40 --seed 7 \
    --softmax-feedback --softmax-feedback-mode one-sided-servo

# GGUF backend works identically (materializes an F32 copy of the embedding
# table, ~671 MB, lazily on the first feedback step)
cargo run --release -- --cpu \
    --model-dir models/gemma-3-270m-it-qat-GGUF \
    --fallback-dir models/gemma-3-270m-it \
    -p "Once upon a time" -n 40 --seed 7 \
    --softmax-feedback --softmax-feedback-mode entropy-servo
```
