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
low-entropy loops). Retained as a fallback and for comparison.

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
  temperatures; blending the fed-back input with the sampled token's
  embedding.

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

# threshold heuristic
cargo run --release -- --cpu -p "Once upon a time" -n 40 --seed 7 \
    --softmax-feedback --softmax-feedback-mode heuristic \
    --softmax-feedback-band 1.0 --softmax-feedback-decay 0.8

# GGUF backend works identically (materializes an F32 copy of the embedding
# table, ~671 MB, lazily on the first feedback step)
cargo run --release -- --cpu \
    --model-dir models/gemma-3-270m-it-qat-GGUF \
    --fallback-dir models/gemma-3-270m-it \
    -p "Once upon a time" -n 40 --seed 7 \
    --softmax-feedback --softmax-feedback-mode entropy-servo
```
