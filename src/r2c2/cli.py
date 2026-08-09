"""r2c2 CLI.

    r2c2 models                    # models with prices on file, and their floors
    r2c2 estimate --model ...      # price an N-sample check, no API calls
    r2c2 check --model ...         # estimate -> sample -> score, the full loop
"""

from __future__ import annotations

import argparse
import json
import sys

from .check import check
from .economics import cost_floor, estimate, required_context
from .pricing import PRICES, PRICES_AS_OF


def _cmd_models(_args) -> int:
    print(f"prices per 1M tokens, as of {PRICES_AS_OF}\n")
    header = f"{'model':<42} {'in':>7} {'cached':>7} {'out':>7} {'write':>7} {'floor(6)':>9}"
    print(header)
    print("-" * len(header))
    for name, price in sorted(PRICES.items()):
        cached = f"{price.cached:.2f}" if price.cached is not None else "-"
        write = f"{price.cache_write:.2f}" if price.cache_write is not None else "-"
        print(
            f"{name:<42} {price.input:>7.2f} {cached:>7} {price.output:>7.2f} "
            f"{write:>7} {cost_floor(6, price):>8.2f}x"
        )
    print("\ncached '-' means no published cached rate: hits bill at full input price.")
    return 0


def _cmd_estimate(args) -> int:
    est = estimate(
        context_tokens=args.context_tokens,
        output_tokens=args.output_tokens,
        model=args.model,
        question_tokens=args.question_tokens,
        n_samples=args.samples,
    )
    needed = required_context(
        output_tokens=args.output_tokens,
        model=args.model,
        question_tokens=args.question_tokens,
        n_samples=args.samples,
        threshold=args.threshold,
    )
    if args.json:
        print(json.dumps(
            {**est.__dict__, "threshold": args.threshold,
             "within_threshold": est.within(args.threshold),
             "required_context_tokens": needed},
            indent=2,
        ))
    else:
        print(est.summary())
        if est.within(args.threshold):
            print(f"threshold {args.threshold:.2f}x -> worth scoring")
        elif needed is None:
            print(
                f"threshold {args.threshold:.2f}x -> unachievable: the floor is "
                f"{est.floor:.2f}x at any context size. Lower --samples or raise the threshold."
            )
        else:
            print(
                f"threshold {args.threshold:.2f}x -> expensive here; clears from "
                f"~{needed:,.0f} context tokens at this output length"
            )
    return 0 if est.within(args.threshold) else 1


def _cmd_check(args) -> int:
    result = check(
        args.model,
        args.context_file.read(),
        args.question,
        n_samples=args.samples,
        threshold=args.threshold,
        expected_output_tokens=args.output_tokens,
        force=args.force,
        on_call=None if args.json else lambda c: print(
            f"  call {c.call}: prompt={c.prompt_tokens} cached={c.cached_tokens} "
            f"-> {c.answer[:70]}"
        ),
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.sampled else 1

    print(f"estimate: {result.estimate.summary()}")
    if not result.sampled:
        print(
            f"estimated {result.estimate.multiplier:.2f}x is above the "
            f"{args.threshold:.2f}x threshold — skipped (rerun with --force to sample anyway)."
        )
        return 1

    print(
        f"\nconfidence {result.confidence:.2f} (min {min(result.scores):.2f}), "
        f"{result.distinct_answers}/{args.samples} distinct answers, "
        f"measured {result.measured_multiplier:.2f}x vs one call"
    )
    if result.confidence < 0.6:
        print("low consistency — treat this answer as unreliable and route it for review.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="r2c2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list models with prices on file").set_defaults(fn=_cmd_models)

    p_est = sub.add_parser("estimate", help="price an N-sample check without calling any API")
    p_est.add_argument("--model", required=True, choices=sorted(PRICES))
    p_est.add_argument("--context-tokens", type=float, required=True)
    p_est.add_argument("--question-tokens", type=float, default=50)
    p_est.add_argument("--output-tokens", type=float, default=300)
    p_est.add_argument("--samples", type=int, default=6)
    p_est.add_argument("--threshold", type=float, default=2.0,
                       help="max acceptable multiplier (exit 1 above it)")
    p_est.add_argument("--json", action="store_true")
    p_est.set_defaults(fn=_cmd_estimate)

    p_chk = sub.add_parser("check", help="estimate, then sample and score if cheap enough")
    p_chk.add_argument("--model", required=True)
    p_chk.add_argument("--context-file", type=argparse.FileType(), required=True)
    p_chk.add_argument("--question", required=True)
    p_chk.add_argument("--samples", type=int, default=6)
    p_chk.add_argument("--output-tokens", type=float, default=300,
                       help="expected answer length for the pre-call estimate")
    p_chk.add_argument("--threshold", type=float, default=2.0)
    p_chk.add_argument("--force", action="store_true", help="sample even above the threshold")
    p_chk.add_argument("--json", action="store_true")
    p_chk.set_defaults(fn=_cmd_check)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
