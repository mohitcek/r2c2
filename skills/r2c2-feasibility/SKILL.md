---
name: r2c2-feasibility
description: >
  Decide whether black-box consistency scoring (sample the same LLM prompt N times, check
  the answers for contradictions) is affordable for a given workload, and at what context
  size it becomes affordable. Use this skill whenever someone asks what N-sample
  confidence / consistency / self-consistency scoring would cost, whether prompt caching
  makes it cheap enough, at what context length scoring "pays off", whether to add an
  uncertainty gate to an agent or RAG pipeline, or anything combining "prompt caching"
  with "sampling" or "black-box UQ" — even if they don't name r2c2. Answers are computed
  by calling the r2c2 package; never estimate a multiplier from memory.
---

# r2c2 feasibility

The folk assumption is that N samples cost N×. Under prompt caching they don't: the
samples are byte-identical prompts, so calls 2..N read the context at the cached rate and
only the answers regenerate. The actual multiplier depends on context size, output length,
and the provider's cached / write / output prices — and it has a **floor** below which no
amount of context helps. This skill computes the multiplier for the caller's numbers and
returns a verdict, using `r2c2.estimate` and `r2c2.required_context`.

Every multiplier is **relative**: scored request ÷ unscored request. Scoring always costs
more than not scoring. Say this every time you report one.

## Procedure

### 0. Ground truth first

Run `python scripts/feasibility.py --check-env` (from this skill's directory). It confirms
`r2c2` imports and prints `prices_as_of` and the known models. If it fails, stop and tell
the user to `pip install r2c2`. Always state how old the prices are in your answer — they
are hardcoded and dated, and cached rates move.

### 1. Gather four inputs

| Input | Ask for | Default if unknown |
|---|---|---|
| `--model` | one of the known models from step 0 | none — ask; an unknown model gets a clean "no price on file" error, report it as such |
| `--context` | cached context tokens per call (C) | estimate from text: characters ÷ 4 |
| `--output` | expected answer length in tokens (O) | 300 (typical short answer); ask if the workload is long-form |
| `--threshold` | the caller's acceptable multiplier | 2.0 |

`--samples` (N) defaults to 6 and `--question` (Q, the uncached tail) to 50; change them
only if the caller says otherwise.

If the caller asks "as context grows" / "at what size does this pay off", use
`--sweep 1000,5000,20000,100000` (or their sizes) instead of `--context` — it prints the
multiplier at each size plus the crossover point in one call.

### 2. Run it, don't recompute it

```
python scripts/feasibility.py --model <m> --context <C> --output <O> --threshold <T>
```

Read the JSON. The fields that matter: `multiplier`, `floor`, `verdict`
(`affordable` | `not_yet` | `never_at_this_threshold`), `required_context_tokens`,
`caveats`. Exit code is 0 when within threshold, 1 when not, 2 on unknown model.

### 3. Report in this shape

1. **Verdict + the two numbers**: "N samples cost **M×** one call (naive guess N×); this
   provider's floor is **F×**." Then the verdict line from `reason`.
2. **If `not_yet`**: give `required_context_tokens` — "clears T× from about X context
   tokens at this output length." That is the actionable answer.
3. **If `never_at_this_threshold`**: say the ceiling is at or below the floor, so no
   context size gets there; the levers are fewer samples or a provider with a cheaper
   cached rate. Offer the multiplier at N=3 if useful (floor ≈ 1 + 2c).
4. **Always include** the relative-cost sentence and `prices_as_of`. Include every item
   in `caveats` that applies — the script only emits the relevant ones.
5. **If affordable and they want to actually score**: point at `r2c2.check(model,
   context, question, threshold=T)` — it estimates, skips if over threshold, otherwise
   samples (call 1 warms the cache, 2..N ride it) and scores with UQLM's
   noncontradiction scorer. Needs the provider API key and the `scoring` extra.

### When to say no

Don't recommend scoring, and say why, when:

- the prompt is short (below the provider's minimum cacheable prefix — 512 tokens on
  Claude Opus 5, 1,024 on OpenAI, 4,096 on Gemini) — nothing caches, it's a true N×;
- the workload is short-prompt / long-answer — output never caches and is priced k× input,
  so the multiplier sits near N (the script's caveat flags this when k·O > C);
- the provider publishes no cached rate for that model — hits bill at full price, pinned
  at N (Qwen and Llama on Together in the measured set);
- the threshold is at or below the floor.

## Interpreting the numbers

The multiplier is a line in the cacheable share f: N when nothing caches, the floor
`w + (N−1)c` when everything does. Floors: 1.50× at a 90% cached discount, 1.75× on
Anthropic (1.25× cache write), N where there's no cached price. Output tokens dilute f
k× faster than context tokens build it — on GPT-5.6 Terra one output token costs ~6
context tokens of headroom. Gemini caches in 8,192-token blocks, so its multiplier
oscillates rather than descending smoothly.

For the derivation, the measured six-model table, per-provider quirks, and the
operational rules (fire sample 1 alone, keep samples byte-identical, temperature > 0),
read `references/cost-model.md`.

Consistency scores rank responses; they are not calibrated probabilities, and consistency
is not correctness — a model that hedges identically N times scores 1.00.
