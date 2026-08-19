# The cost model behind the verdict

Everything `scripts/feasibility.py` reports comes from one closed form, implemented in
`r2c2.economics`. Read this when you need to explain *why* a verdict came out the way it
did, or when the caller's situation isn't one the script covers directly.

## The equation

For N samples of the same prompt, cost relative to one unscored call:

```
M = N − f · (N − w − (N−1)·c)          f = C / (C + Q + k·O)

C  cached context tokens          c  cached rate ÷ input rate   (~0.1 where a cached price exists; 1.0 if none)
Q  uncached input per call        w  cache-write rate ÷ input   (1.0 most places; 1.25 on Anthropic)
O  output tokens per call         k  output price ÷ input price (6 Terra · 5 Opus, Gemini · 2 DeepSeek)
```

`f` is the share of one request's *cost* that can ride the cache. `M` is a straight line
in it: `N` when nothing caches (f = 0), the **floor** `w + (N−1)c` when everything does
(f = 1). Six measured models land on that line to within 0.005.

The floor is approached from above and never crossed. For six samples at a 90% cached
discount it is 1.50×; Anthropic's 1.25× cache write lifts it to 1.75×. Below the floor no
context size helps — only fewer samples or a cheaper cached rate.

Inversion (what `required_context()` computes): pick a ceiling M*, then
`f = (N − M*) / (N − floor)` and `C = f / (1 − f) · (Q + k·O)`. That is the smallest
context at which scoring clears the ceiling at that output length.

## The rule that falls out: C ≫ k·O

Output tokens enter `f` multiplied by `k` — they are priced k× higher and never cache. On
GPT-5.6 Terra (k = 6) one output token dilutes the cacheable share as fast as six context
tokens build it; at a 1.8× ceiling each extra output token needs ~84 more context tokens
just to hold the multiplier steady. Long context helps; long answers hurt, k times faster.
Short prompts with long answers are the regime where the old "N samples = N× cost"
objection is simply correct.

## Measured (31 Jul 2026, ~21k-token context, 6 samples, cache warm)

| Model | Cache hit | 6 samples vs 1 call | Floor |
|---|---|---|---|
| gpt-5.6-terra (OpenAI) | 100% | 1.55× | 1.50× |
| claude-opus-5 (Anthropic) | 99.8% | 1.81× | 1.75× |
| DeepSeek-V4-Pro (Together) | 99.0% | 1.86× | 1.57× |
| gemini-3.6-flash (Google) | 78.2% | 2.52× | 1.50× |
| Qwen3.6-Plus (Together) | 0% | 6.00× | 6.00× |
| Llama-3.3-70B-Turbo (Together) | 99.4% | 6.00× | 6.00× |

Source: `tests/data/receipt_uq.json` on `main`; the full sweep (1.4k → 440k tokens) and
all raw receipts are on the `medium-blog-post` branch.

## Provider quirks that change the verdict

- **A cache hit is not a discount.** Llama on Together cached 99.4% of the prompt and
  saved $0 — Together publishes a cached rate per model, and Llama has none. With no
  cached price, `c = 1` and the multiplier is pinned at N however much caches.
- **Anthropic** bills a one-time 1.25× cache write on the cold call (floor 1.75×, not
  1.50×), and is the only provider of the four that reports the write at all. Its
  `input_tokens` is the *uncached remainder* only; total prompt = input + cache_creation
  + cache_read.
- **Gemini 3 Flash** caches in 8,192-token blocks, so the remainder past the last block
  bills full price and the multiplier oscillates with context size (2.07× at 28k, back
  up to 2.31× at 50k, 1.63× at 84k). In our runs it cached nothing below ~7k tokens.
- **Minimum cacheable prefix** differs: 512 tokens on Claude Opus 5, 1,024 on OpenAI,
  4,096 on Gemini 3.x. Below it, zeros with no error.
- **Prices are dated.** `r2c2.PRICES_AS_OF` says when; the script prints it. Cached
  rates move more often than list prices — verify before quoting a multiplier externally.

## Operational notes (they decide whether you actually get the multiplier)

- **Fire sample 1 alone, then fan out.** A cache entry becomes readable only after the
  first response starts streaming; N concurrent identical requests all miss.
- **Keep the samples byte-identical**, with anything volatile (timestamps, request IDs)
  placed *after* the context. One early differing byte invalidates everything downstream.
- **Sample at temperature > 0.** Identical samples give a degenerate, falsely perfect
  consistency score.
- **Consistency is not correctness.** A model that hedges identically six times scores
  1.00. Scores rank; they are not calibrated probabilities — tune thresholds on your own
  traffic.
