# r2c2

**R**euse **C**ontext, **R**echeck **C**onsistency — ask the model again at cache prices, and score the agreement.

The most portable way to catch an LLM being unreliable is to sample the same prompt N times and check the answers for contradictions. The objection is always cost: "six samples means paying six times." It doesn't. Consistency sampling resends byte-identical prompts — the best-case workload for prompt caching. Measured across 6 models / 4 providers with a ~21k-token context and 6 samples per request:

| Model | Cache hit | Cost of 6 samples vs 1 call |
|---|---|---|
| gpt-5.6-terra (OpenAI) | 100% | **1.55×** |
| claude-opus-5 (Anthropic) | 99.8% | **1.81×** |
| DeepSeek-V4-Pro (Together) | 99.0% | **1.86×** |
| gemini-3.6-flash (Google) | 78.2% | **2.52×** |
| Qwen3.6-Plus (Together) | 0% | 6.00× |
| Llama-3.3-70B-Turbo (Together) | 99.4% (no cached price) | 6.00× |

**Read the last two rows first.** Qwen cached nothing. Llama cached 99.4% of the prompt and still saved nothing, because Together publishes cached rates per model and Llama has none — a cache hit is not a discount. Before trusting any multiplier on a provider you haven't measured, check that its price sheet has a cached input line. Without one, consistency sampling costs the full N× no matter what the cache telemetry reports.

And the score catches real failures: on a question the test policy doesn't answer, Qwen went No-Yes-No-Yes-No-Yes across six samples (noncontradiction 0.40) and Gemini cited regulations that appear nowhere in the prompt. On a question the policy answers directly, every model scores 0.99+.

r2c2 turns that finding into a workflow: **estimate → sample → score**. The measurements, experiment scripts, receipts, and figures behind these numbers live on the [`medium-blog-post`](../../tree/medium-blog-post) branch.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync                        # core: cost math + CLI, zero dependencies
uv sync --extra providers      # + OpenAI / Anthropic / Google SDKs for sampling
uv sync --extra scoring        # + UQLM noncontradiction scoring (pulls torch, large)
```

## Quickstart

**1. Price the check first — pure arithmetic, no API calls:**

```python
from r2c2 import estimate, required_context

est = estimate(context_tokens=21_000, output_tokens=40, model="gpt-5.6-terra")
est.multiplier      # 1.56 — six samples cost 1.56x one call, not 6x
est.within(2.0)     # True — cheap enough to score

# or invert it: pick a ceiling, get the context size where scoring clears it
required_context(output_tokens=40, model="gpt-5.6-terra", threshold=1.8)
# 4060.0 — under ~4k context tokens, six samples cost more than 1.8x
required_context(output_tokens=40, model="claude-opus-5", threshold=1.6)
# None — Anthropic's floor is 1.75x; no context size gets there. Lower n_samples.
```

Same thing from the shell (`r2c2 models` lists supported models); above the
threshold, `estimate` tells you where the check would start clearing it:

```bash
uv run r2c2 estimate --model gpt-5.6-terra --context-tokens 2000 --output-tokens 40 --threshold 1.8
# threshold 1.80x -> expensive here; clears from ~4,060 context tokens at this output length
```

Every estimate prints the rate ratios that produced its floor, so a provider with no
cached line names itself instead of looking like a rounding error:

```text
gpt-5.6-terra          ... floor 1.50x at c=0.10
claude-opus-5          ... floor 1.75x at c=0.10, w=1.25     <- 1.25x cache write
Llama-3.3-70B-Turbo    ... floor 6.00x at c=1.00 — no cached rate
```

**2. Run the whole loop with one call.** `check` estimates, refuses if the surcharge is unacceptable, and otherwise collects N samples (call 1 warms the cache, calls 2–N ride it) and scores their agreement:

```python
from r2c2 import check

result = check("gpt-5.6-terra", context, question, threshold=2.0)
result.sampled              # False -> too expensive, no API calls were made
result.confidence           # 0.99 = answers agree; 0.40 = coin flip
result.measured_multiplier  # what the N samples actually cost vs one call
```

Low confidence → route to a human, a retrieval retry, or a bigger model. Same loop from the shell:

```bash
export OPENAI_API_KEY=...
uv run r2c2 check --model gpt-5.6-terra --context-file policy.txt \
    --question "Are they eligible for a refund?" --threshold 2.0
```

**3. Or use the pieces directly:**

```python
from r2c2 import sample, confidence

calls = sample("gpt-5.6-terra", context, question, n_samples=6)
confidence([c.answer for c in calls])   # local NLI, no extra API calls
```

## Use it from an agent

`skills/r2c2-feasibility/` is an Agent Skill (a `SKILL.md` with a script and a
reference) that lets a coding agent answer "is consistency scoring affordable here, and
from what context size?" by calling this package rather than guessing. Copy the folder
into your agent's skills directory — `~/.claude/skills/` for Claude Code — and ask
something like *"would 6-sample scoring be affordable for a 30k-token RAG agent on Claude
Opus 5 with 200-token answers?"*. The skill runs `scripts/feasibility.py`, which returns a
JSON verdict (`affordable` / `not_yet` + the crossover context size /
`never_at_this_threshold` + the floor) with the caveats that apply.

## The cost model

For N samples of the same prompt:

```
M = N − f · (N − w − (N−1)·c)        f = C / (C + Q + k·O)

C  cached context tokens        c  cached ÷ input rate   (~0.1; 1.0 if no cached price)
Q  uncached input per call      w  write ÷ input rate    (1.0; 1.25 on Anthropic)
O  output tokens                k  output ÷ input price
```

`f` is the cost-weighted share of the request that rides the cache; `M` is a straight line in it — N when nothing caches, the floor `w + (N−1)c` when everything does. The tests validate this against a shipped receipt of live measured runs (`tests/data/receipt_uq.json`) to within a few hundredths.

## Layout

```text
src/r2c2/
  pricing.py       published prices (dated — re-verify before quoting)
  economics.py     cost_multiplier / cost_floor / estimate / required_context
  providers.py     4 API runners, cached-token counts normalised
  scoring.py       UQLM noncontradiction wrapper (lazy import)
  check.py         check(): the estimate -> sample -> score loop
  cli.py           r2c2 models | estimate | check
tests/             offline; validates the closed form against a measured receipt
skills/r2c2-feasibility/   agent skill: SKILL.md + scripts/feasibility.py + references/
```

## Things that will bite you

- **Fire sample 1 alone, then fan out.** A cache entry is readable only after the first response begins; N concurrent identical calls all miss (`sample` is sequential for this reason).
- **No cached rate means no saving** (see the table above). `r2c2` encodes this: a model with no cached price on file gets `c = 1`, so `estimate` pins the multiplier at N and prints `c=1.00 — no cached rate` rather than promising a discount the invoice won't show.
- **Gemini 3 Flash has a caching dead zone** at ~9k–17k prompt tokens and caches in 8,192-token blocks above it; below ~7k it caches nothing at all.
- **Anthropic's `input_tokens` is the uncached remainder only** — total prompt size is `input + cache_creation + cache_read`. It also charges a one-time 1.25× cache write, putting its floor at 1.75× rather than 1.50×.
- **Scores are rankings, not probabilities.** Consistency is not correctness: a model that hedges identically six times scores 1.00. Tune thresholds on your own traffic.
- **You need C ≫ k·O.** Output never caches and is priced k× input, so long answers erode the saving k× faster than long contexts build it. Short prompts + long answers ≈ the old 6× objection, unchanged.

## Roadmap

- An MCP server exposing `estimate` / `required_context` / `check` as tools, for clients that don't read `SKILL.md` (`CheckResult.to_dict()` is the intended tool payload).
- Additional UQLM black-box scorers (semantic entropy, exact-match) behind the same interface.

## Provenance

Grew out of the blog post [*"Prompt Caching Makes Self-Consistency Cheap for Long-Context LLMs"*](https://medium.com/@mohitsinghchauhan/prompt-caching-makes-self-consistency-cheap-for-long-context-llms-c611ba4f5237) — the experiment scripts, raw JSON receipts, and figures are on the [`medium-blog-post`](../../tree/medium-blog-post) branch. Built on [UQLM](https://github.com/cvs-health/uqlm). Prices and model IDs are as of 2026-07-31 — verify yours before trusting any multiplier here. MIT licensed.
