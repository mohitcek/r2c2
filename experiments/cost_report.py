"""Turn a receipt into the measured cost multipliers for consistency scoring.

The folk objection is "N samples = N x cost". This works out the real number from the
measured token counts: 1 cold call plus (N-1) cache hits, output included.

    python experiments/cost_report.py --receipt receipts/receipt_uq.json
"""

import argparse
import json
from dataclasses import asdict

from r2c2 import PRICES, PRICES_AS_OF, measured_costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default="receipts/receipt.json")
    parser.add_argument("--out", default="receipts/cost_model.json")
    args = parser.parse_args()

    with open(args.receipt) as fh:
        receipt = json.load(fh)

    n_calls = receipt["calls_per_model"]
    rows = []

    for run in receipt["runs"]:
        model = run["model"]
        if model not in PRICES or not run["calls"]:
            continue
        price = PRICES[model]
        calls = run["calls"]

        prompt = calls[0]["prompt_tokens"]
        # Steady-state hit level, not call 2's — Llama on Together takes four calls
        # to warm up, and using call 2 would understate it.
        cached = max(c["cached_tokens"] for c in calls)
        written = max(c["cache_write_tokens"] for c in calls)
        out_tokens = sum(c["output_tokens"] for c in calls) / len(calls)

        single, scored = measured_costs(prompt, cached, written, out_tokens, price, n_calls)
        naive = n_calls * single
        rows.append(
            {
                "provider": run["provider"],
                "model": model,
                "prompt_tokens": prompt,
                "steady_cached_tokens": cached,
                "hit_share": round(cached / prompt, 4),
                "mean_output_tokens": round(out_tokens, 1),
                "cost_single_unscored": round(single, 6),
                "cost_scored_naive_assumption": round(naive, 6),
                "cost_scored_with_caching": round(scored, 6),
                "multiplier_naive": round(naive / single, 2),
                "multiplier_actual": round(scored / single, 2),
                "scored_per_1k_requests": round(scored * 1000, 2),
                "naive_per_1k_requests": round(naive * 1000, 2),
            }
        )

    rows.sort(key=lambda r: r["multiplier_actual"])

    print(f"{n_calls} samples per scored request | prices as of {PRICES_AS_OF}\n")
    print(f"{'model':<38} {'assumed':>8} {'actual':>8} {'$/1k naive':>11} {'$/1k cached':>12}")
    print("-" * 82)
    for r in rows:
        print(
            f"{r['model']:<38} {r['multiplier_naive']:>7.1f}x {r['multiplier_actual']:>7.2f}x "
            f"{r['naive_per_1k_requests']:>10.2f} {r['scored_per_1k_requests']:>11.2f}"
        )

    with open(args.out, "w") as fh:
        json.dump(
            {
                "prices_as_of": PRICES_AS_OF,
                "prices_per_million": {name: asdict(p) for name, p in PRICES.items()},
                "samples_per_scored_request": n_calls,
                "receipt": args.receipt,
                "question_name": receipt.get("question_name"),
                "assumption": "cache warm across the N samples; steady-state hit level; "
                "measured token counts",
                "models": rows,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
