# Prompt Caching Makes Self-Consistency Cheap for Long-Context LLMs

## Consistency-based scoring resends the same prompt — exactly what prompt caching rewards. The bigger your context, the smaller the surcharge for checking your model's answers. Measured across six models and four providers.

*I work on [UQLM](https://github.com/cvs-health/uqlm), the open-source uncertainty-quantification library used here. Every number below comes from live API calls made on July 30–31, 2026; the code and raw JSON are linked at the end.*

---

The most portable way to catch an LLM being unreliable has been known for years: ask the same question several times and check whether the answers agree. No logprobs needed, no labeled data, no judge model. It works on any API you can call twice.

Almost nobody runs it in production, and the objection is always the same sentence: *"sampling six responses means paying six times."*

I measured that across six models. Where prompt caching works, six samples cost 1.55 to 2.5 times one call. Not six.

One thing to be explicit about before the numbers, because it's the difference between a result and a bait headline: these are **relative** figures. A scored request still costs more than an unscored one — your bill goes up, not down. The claim is narrower: the surcharge for scoring is around 55%, not the 500% everyone assumes. The tables carry absolute dollars so you can check that framing rather than take it.

## The perfect caching workload

The reason the multiplier collapses is mechanical. The API sees your request as one token sequence — context first, question after — and prompt caching matches that sequence from byte zero forward, serving the matched span at roughly 90% off the input price. Change one early byte and everything after it bills full price again.

Now look at what consistency sampling does: it sends the same prompt, byte for byte, N times. Not similar prompts. Identical ones. The first sample pays full price and writes the cache; samples two through six read the context at a tenth of the rate, and only the answers regenerate — which are the thing you're checking for contradictions anyway. In the workloads where confidence matters most, RAG over long documents and agents with big tool contexts, input tokens dominate the bill. So the "6×" mostly evaporates.

*[Figure: fig4_prefix.png — how a cold call and a warm call bill differently]*

There is fine print, and it bites. Anthropic charges a one-time 1.25× premium to write the cache, and is the only provider of the three that reports the write at all. Gemini caches in 8,192-token blocks, and in my runs cached nothing between roughly 9k and 17k tokens. Minimum cacheable sizes differ too: 512 tokens on Claude Opus 5, 1,024 on OpenAI, 4,096 on Gemini.

## The receipt

The setup: a ~21k-token airline policy as context, one question after it, the same call sent six times per model. For each call I recorded the provider's cached-token count — four providers, four different field names for the same fact — then scored the six answers with UQLM's `noncontradiction` scorer, an NLI model that checks each answer against the other five. Scoring runs locally, after the fact, and costs no extra API calls.

The folk assumption says six samples cost 6×. Measured:

*[Figure: fig1_wide_1600x900.png — the multiplier, measured]*

| Model | Cache hit | Surcharge for scoring | Per 1k requests: unscored → scored |
|---|---|---|---|
| gpt-5.6-terra (OpenAI) | 100% | **+55%** (1.55×) | $52 → $80 *(naive guess: $310)* |
| claude-opus-5 (Anthropic) | 99.8% | +81% (1.81×) | $160 → $289 *($959)* |
| DeepSeek-V4-Pro (Together) | 99.0% | +86% (1.86×) | $41 → $76 *($246)* |
| gemini-3.6-flash (Google) | 78.2% | +152% (2.52×) | $32 → $80 *($190)* |
| Qwen3.6-Plus (Together) | 0% | +500% (6.00×) | $18 → $105 |
| Llama-3.3-70B-Turbo (Together) | 99.4% | +500% (6.00×) | $19 → $113 |

The failures tell you as much as the wins. Qwen reported zero cache hits across all six calls, so it sits at exactly 6× — the objection holds there, unchanged. Llama cached 99.4% of the prompt and saved nothing, because Together publishes a discounted cached rate for some models and not this one. A cache hit is an infrastructure fact; a discount is a pricing decision. You need both.

## Cheaper as context grows

The “long-context” in the title is a claim, so I measured it: a sweep from 1.4k up to 440k tokens, three calls at each size, every size on its own cold prefix so nothing inherits a warm cache from the run before.

*[Figure: fig5_curve.png — the multiplier vs context size]*

Every curve falls and flattens. The flattening is an asymptote, not a trend that keeps going — OpenAI reads 1.51×, 1.51×, 1.50×, 1.50× from 55k out to 440k tokens. Eight times more context buys a hundredth. Past the floor, the only ways down are fewer samples (three samples floor at 1.20×) or a deeper cached discount.

Two providers put their own spin on the shape. Anthropic's floor sits at 1.75× rather than 1.50×, because that 1.25× cache write is paid on every cold call. Gemini takes the scenic route: the 8,192-token blocks mean its multiplier oscillates as the prefix crosses each boundary — 2.07× at 28k, back up to 2.31× at 50k, down to 1.63× at 84k — on its way to the same floor as everyone else.

And below about 7k tokens Gemini cached nothing at all, sitting at a flat 6.00×. That's the honest boundary of the whole argument: **if your prompts are small, the old objection is simply correct.**

## One equation

All of it collapses into a single line. For N samples of the same prompt:

```
M = N − f · (N − w − (N−1)·c)        f = C / (C + Q + k·O)

C  cached context tokens        c  cached ÷ input rate   (~0.1; 1.0 if no cached price)
Q  uncached input per call      w  write ÷ input rate    (1.0; 1.25 on Anthropic)
O  output tokens                k  output ÷ input price  (6 Terra · 5 Opus, Gemini · 2 DeepSeek)
```

`f` is the share of your request that can ride the cache, weighted by cost, and `M` is a straight line in it: N when nothing caches, the floor when everything does. Every model I measured lands on that line to within 0.005.

The cost-weighting is the part worth internalizing. Output tokens enter `f` multiplied by `k`, because they're priced k times higher and never cache. On Terra, one output token dilutes your cacheable share as fast as six context tokens build it — which explains most of the table above. DeepSeek sits at 1.86× despite a 99% hit rate because it wrote 656 tokens of reasoning per answer. Llama lands at exactly 6.00× with f = 0.99 because no cached price means c = 1, and the line has nowhere to go.

The rule that falls out: **you need C ≫ k·O**. Long context helps. Long answers hurt, k times faster.

*[Figure: fig6_heatmap.png — the multiplier over context × output]*

For your own stack it's arithmetic, not a benchmark:

```python
# pip install r2c2 — Reuse Context, Recheck Consistency
from r2c2 import estimate, required_context

estimate(context_tokens=21_000, output_tokens=40, model="gpt-5.6-terra").multiplier
# 1.56 — matches the 1.55 measured

required_context(output_tokens=40, model="gpt-5.6-terra", threshold=1.8)
# 4060 — under ~4k context tokens, six samples cost more than 1.8×
```

The second call is the same line read backwards: pick a cost ceiling, get back the context size where scoring starts clearing it — or `None` when no context size ever will, because the ceiling sits at or below that provider's floor.

## What the surcharge buys

Cheap sampling only matters if the score catches something, so the test question wasn't a softball. The policy says non-refundable fares can be refunded within 24 hours of booking. It says nothing about who cancelled. So: *"A customer booked a non-refundable fare 30 hours ago. The airline then cancelled the flight. Are they eligible for a refund?"*

On a control question the policy answers directly, all six models score 0.99+. The wording changes every time, the verdict never does — rewording is what sampling is; it isn't disagreement.

On the airline question, from the very same six calls as the cost receipt, things come apart. Qwen answered No, Yes, No, Yes, No, Yes — a coin flip on a refund decision, scored 0.40. Gemini split 4–2, and its yes answers cite "Department of Transportation regulations" that appear nowhere in the prompt. Terra and DeepSeek each split 5–1. Claude Opus 5 said yes all six times, having settled — reasonably, but without any support in the text — on a reading where the 24-hour rule only governs customer-initiated cancellations. And Llama scored a perfect 1.00 by hedging identically six times: "may be considered an exception… may be eligible." Refusing to commit, consistently, is also a signal. Just not one this scorer flags.

*[Figure: fig2_split.png — 0.99 vs 0.40, side by side]*

All thirty-six answers are fluent, confident, and cite the policy. Nothing in any single response tells you the model would say the opposite on the next sample. The 0.40 is the only visible symptom — and in an input-heavy workload it now costs about half of one extra call.

The splits also move between runs. On another run Terra went 3–3 and Gemini refused all six times. You can't pre-test your way out of that, which is exactly the argument for scoring at runtime instead of auditing once, offline.

## The recipe

If you already sample N responses, the scoring itself is free — `.score()` runs post-hoc on answers you have:

```python
from uqlm import BlackBoxUQ

# six answers you already collected; calls 2-6 hit the prompt cache
scorer = BlackBoxUQ(scorers=["noncontradiction"], use_best=False)
result = scorer.score(responses=[answers[0]], sampled_responses=[answers[1:]])
confidence = result.to_df()["noncontradiction"][0]  # 0.99 = agrees; 0.40 = coin flip
```

The scorer runs a local NLI model (`microsoft/deberta-large-mnli`), so there's no judge-model cost on top. Route low scores to a human, a retrieval retry, or a bigger model.

Three details decide whether you actually get the 1.55×. Fire the first call alone and fan out after it — a cache entry only becomes readable once the first response starts streaming, so six concurrent identical requests all miss. Keep the samples byte-identical, with anything volatile placed after the context; one interpolated timestamp invalidates everything downstream. And sample at a temperature above zero, or six identical answers will hand you a perfect score that means nothing.

The caveats I'd want to be held to: the multiplier assumes the cache stays warm across your samples (TTLs run about five minutes and up). Scores are rankings, not probabilities — 0.40 doesn't mean "40% likely correct," and consistency isn't correctness; a model that's confidently, repeatably wrong scores 1.00. The six models here are different tiers at different prices, so this is caching economics, not a model ranking. And pricing is as of July 31, 2026, per provider and per model. Verify yours.

## The point

Black-box UQ was always the most deployable form of hallucination detection: no logprobs, no labels, works everywhere. The one respectable argument against it was cost, and that argument quietly expired when providers shipped 90%-off cached reads — because consistency sampling resends the same bytes by construction.

- Cache the reading.
- Re-sample the answer.
- Score the agreement.
- Route the 0.40s to a human.

On the platforms most production traffic runs on, that loop adds about half the cost of one extra call.

Don't copy my numbers, though. They're a function of my context size, my providers, and July 2026 pricing — a 2k-token prompt caches nothing on Gemini, and a chatbot with short prompts and long answers won't see 1.55×. The equation takes thirty seconds; the repo takes four API keys and about two dollars.

But the direction of travel favors this technique, and the reason is agents. Agent contexts are the fastest-growing prompts in production — every tool call appends, every turn re-sends the whole history — and even with real effort spent on compaction and pruning, the absolute numbers keep climbing. That growth doesn't erode this math. It walks you down the curve. Providers built prompt caching because agents re-send huge prefixes; consistency scoring rides the same infrastructure for free.

The workloads where verification matters most are becoming the ones where it's cheapest.

---

*Code, receipts, and figures — the r2c2 repo (Reuse Context, Recheck Consistency): https://github.com/mohitcek/r2c2 · Interactive version of these tables: [claude.ai/code/artifact/67ab1363…](https://claude.ai/code/artifact/67ab1363-692c-40a9-be3a-abc9fe7cc0b4) · UQLM: [github.com/cvs-health/uqlm](https://github.com/cvs-health/uqlm)*
