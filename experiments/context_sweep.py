"""Sweep context size and record how the scoring cost multiplier moves.

As the prompt grows relative to the output, the cost of N samples converges on

    (write_rate / input_rate) + (N-1) * (cached_rate / input_rate)

which is 1.50x for six samples at a 90% cached discount. This measures the approach
rather than asserting it.

    python experiments/context_sweep.py
    python experiments/context_sweep.py --models gpt-5.6-terra --reps 20,100,800
"""

import argparse
import json
import os
from statistics import mean

from policy import QUESTIONS, build_context

from r2c2 import PRICES, cost_floor, measured_costs
from r2c2.providers import spec_for

DEFAULT_REPS = [20, 50, 100, 200, 400, 800]
DEFAULT_MODELS = ["gpt-5.6-terra", "claude-opus-5", "gemini-3.6-flash"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--reps", default=",".join(str(r) for r in DEFAULT_REPS))
    parser.add_argument("--calls", type=int, default=3, help="real calls per size")
    parser.add_argument("--samples", type=int, default=6, help="samples to price for")
    parser.add_argument("--question", default="control")
    parser.add_argument("--out", default="receipts/context_sweep.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    reps_list = [int(r) for r in args.reps.split(",") if r.strip()]
    question = QUESTIONS[args.question]

    rows = []
    for model in models:
        spec = spec_for(model)
        if not os.environ.get(spec.env):
            print(f"-- {model}: skipped, {spec.env} not set")
            continue

        price = PRICES[model]
        print(f"\n-- {model}")
        for reps in reps_list:
            # Own nonce per size. reps=20's context is a literal prefix of reps=50's,
            # so sharing one would leave every size after the first with a warm
            # "cold" call.
            context = build_context(reps, f"sweep-{model}-{reps}")
            try:
                calls = [
                    spec.runner(model, context, question, c)
                    for c in range(1, args.calls + 1)
                ]
            except Exception as exc:
                print(f"   reps={reps:<5} FAILED {type(exc).__name__}: {exc}")
                continue

            prompt = calls[0].prompt_tokens
            cached = max(c.cached_tokens for c in calls)
            written = max(c.cache_write_tokens for c in calls)
            out_tokens = mean(c.output_tokens for c in calls)

            # 3 real calls, but price the N-sample request: the formula needs one
            # cold price and one warm price, not N of each.
            single, scored = measured_costs(
                prompt, cached, written, out_tokens, price, args.samples
            )
            rows.append(
                {
                    "model": model,
                    "reps": reps,
                    "prompt_tokens": prompt,
                    "cached_tokens": cached,
                    "hit_share": round(cached / prompt, 4),
                    "mean_output_tokens": round(out_tokens, 1),
                    "multiplier": round(scored / single, 3),
                    "cost_single": round(single, 6),
                    "cost_scored": round(scored, 6),
                }
            )
            print(
                f"   reps={reps:<5} prompt={prompt:>7}  cached={cached:>7}  "
                f"hit={cached / prompt:>6.1%}  ->  {scored / single:.2f}x"
            )

    floors = {
        m: round(cost_floor(args.samples, PRICES[m]), 3)
        for m in models
        if m in PRICES and PRICES[m].cached is not None
    }

    with open(args.out, "w") as fh:
        json.dump(
            {
                "samples_per_scored_request": args.samples,
                "real_calls_per_point": args.calls,
                "question": args.question,
                "note": "each size uses its own front-of-prompt nonce, so call 1 is "
                "genuinely cold at every point",
                "theoretical_floor": floors,
                "floor_note": "floor = (write_rate/input_rate) + (N-1)*(cached_rate/"
                "input_rate). Anthropic's 1.25x write puts its floor at 1.75x, not 1.50x. "
                "Gemini reaches the same floor as OpenAI but oscillates on the way: it "
                "caches in 8,192-token blocks, so the multiplier depends on where the "
                "prefix falls relative to a boundary.",
                "points": rows,
            },
            fh,
            indent=2,
        )
    print(f"\nfloors: {floors}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
