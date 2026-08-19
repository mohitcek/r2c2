"""Is N-sample consistency scoring affordable here? Computed, not recalled.

Thin CLI over r2c2.estimate / r2c2.required_context. Prints one JSON object so an
agent can read the verdict instead of redoing the arithmetic.

    python feasibility.py --check-env
    python feasibility.py --model gpt-5.6-terra --context 21000 --output 40
    python feasibility.py --model gpt-5.6-terra --context 2000 --output 40 --threshold 1.8
    python feasibility.py --model claude-opus-5 --output 200 --sweep 1000,5000,20000,100000
"""

import argparse
import json
import sys


def check_env():
    try:
        import r2c2
        from r2c2 import PRICES, PRICES_AS_OF
    except ImportError:
        print(json.dumps({"ok": False, "error": "r2c2 is not importable — pip install r2c2"}))
        return 1
    print(json.dumps({
        "ok": True,
        "r2c2_version": r2c2.__version__,
        "prices_as_of": PRICES_AS_OF,
        "models": sorted(PRICES),
    }, indent=2))
    return 0


def verdict_for(est, threshold, needed):
    if est.within(threshold):
        return "affordable", (
            f"{est.n_samples} samples cost {est.multiplier:.2f}x one call "
            f"(floor {est.floor:.2f}x), under the {threshold:.2f}x ceiling."
        )
    if needed is None:
        return "never_at_this_threshold", (
            f"The {threshold:.2f}x ceiling is at or below this provider's floor of "
            f"{est.floor:.2f}x, which is approached from above and never crossed. No "
            f"context size gets there: lower n_samples or pick a provider with a "
            f"cheaper cached rate."
        )
    return "not_yet", (
        f"{est.multiplier:.2f}x now; clears {threshold:.2f}x from about "
        f"{needed:,.0f} context tokens at this output length."
    )


def caveats_for(est, price, model):
    out = [
        "Relative cost: the multiplier compares a scored request to the same request "
        "unscored. Scoring still costs more than not scoring.",
    ]
    if price.cached is None:
        out.append(
            f"{model} has no published cached rate, so cache hits bill at full input "
            f"price and the multiplier is pinned at N regardless of context size."
        )
    k = price.output_ratio
    if k >= 3:
        out.append(
            f"Output is priced {k:g}x input here and never caches: each extra output token "
            f"costs about {k:g} context tokens of headroom (you need C >> k*O)."
        )
    if est.output_tokens * k > est.context_tokens:
        out.append(
            "Output cost exceeds cached-context cost for this request — this is the "
            "short-prompt / long-answer regime where the old N-times objection holds."
        )
    if "gemini" in model.lower():
        out.append(
            "Gemini caches in 8,192-token blocks and, in our runs, cached nothing below "
            "~7k tokens; expect the multiplier to oscillate with context size."
        )
    out.append("Consistency scores rank; they are not calibrated probabilities.")
    return out


def run(args):
    from r2c2 import PRICES, PRICES_AS_OF, cost_floor, estimate, required_context

    if args.model not in PRICES:
        print(json.dumps({
            "ok": False,
            "error": f"no price on file for {args.model!r}",
            "known_models": sorted(PRICES),
        }, indent=2))
        return 2

    price = PRICES[args.model]
    common = dict(model=args.model, question_tokens=args.question, n_samples=args.samples)

    if args.sweep:
        rows = []
        for c in args.sweep:
            e = estimate(context_tokens=c, output_tokens=args.output, **common)
            rows.append({
                "context_tokens": c,
                "multiplier": round(e.multiplier, 3),
                "within_threshold": e.within(args.threshold),
            })
        needed = required_context(output_tokens=args.output, threshold=args.threshold, **common)
        print(json.dumps({
            "ok": True,
            "model": args.model,
            "n_samples": args.samples,
            "output_tokens": args.output,
            "threshold": args.threshold,
            "floor": round(cost_floor(args.samples, price), 3),
            "required_context_tokens": None if needed is None else round(needed),
            "sweep": rows,
            "prices_as_of": PRICES_AS_OF,
        }, indent=2))
        return 0

    est = estimate(context_tokens=args.context, output_tokens=args.output, **common)
    needed = required_context(output_tokens=args.output, threshold=args.threshold, **common)
    verdict, reason = verdict_for(est, args.threshold, needed)

    print(json.dumps({
        "ok": True,
        "model": args.model,
        "inputs": {
            "context_tokens": args.context,
            "output_tokens": args.output,
            "question_tokens": args.question,
            "n_samples": args.samples,
            "threshold": args.threshold,
        },
        "multiplier": round(est.multiplier, 3),
        "naive_multiplier": float(args.samples),
        "surcharge_pct": round(est.surcharge * 100, 1),
        "floor": round(est.floor, 3),
        "within_threshold": est.within(args.threshold),
        "required_context_tokens": None if needed is None else round(needed),
        "cost_per_request_usd": {
            "unscored": round(est.cost_single, 6),
            "scored": round(est.cost_scored, 6),
        },
        "verdict": verdict,
        "reason": reason,
        "caveats": caveats_for(est, price, args.model),
        "prices_as_of": PRICES_AS_OF,
    }, indent=2))
    return 0 if est.within(args.threshold) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--context", type=float, help="cached context tokens (C)")
    ap.add_argument("--output", type=float, default=300, help="expected output tokens (O)")
    ap.add_argument("--question", type=float, default=50, help="uncached input per call (Q)")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=2.0, help="max acceptable multiplier")
    ap.add_argument("--sweep", type=lambda s: [float(x) for x in s.split(",")],
                    help="comma-separated context sizes; prints the multiplier at each")
    args = ap.parse_args(argv)

    if args.check_env:
        return check_env()
    if not args.model or (args.context is None and not args.sweep):
        ap.error("--model and either --context or --sweep are required")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
