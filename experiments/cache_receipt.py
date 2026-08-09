"""Send the same prompt N times to each model and record what the cache did.

    python experiments/cache_receipt.py --nonce run1
    python experiments/cache_receipt.py --providers openai,together --question airline-fault
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from policy import DEFAULT_REPS, QUESTIONS, build_context

from r2c2.providers import PROVIDERS, ModelRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument("--calls", type=int, default=6)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--question", default="control", choices=sorted(QUESTIONS))
    parser.add_argument(
        "--nonce", default="", help="prepended to the prefix to force a cold cache"
    )
    parser.add_argument("--out", default="receipts/receipt.json")
    args = parser.parse_args()

    selected = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in selected if p not in PROVIDERS]
    if unknown:
        sys.exit(f"unknown provider(s): {', '.join(unknown)}")

    context = build_context(args.reps, args.nonce)
    question = QUESTIONS[args.question]
    print(f"prefix: {args.reps} clauses, {len(context)} chars")
    print(f"{args.calls} identical calls per model, question={args.question}\n")

    runs = []
    for name in selected:
        spec = PROVIDERS[name]
        if not os.environ.get(spec.env):
            print(f"-- {name}: skipped, {spec.env} not set")
            continue

        for model in spec.models:
            run = ModelRun(provider=name, model=model)
            runs.append(run)
            print(f"-- {name} / {model}")
            for call in range(1, args.calls + 1):
                try:
                    result = spec.runner(model, context, question, call)
                except Exception as exc:
                    run.error = f"{type(exc).__name__}: {exc}"
                    print(f"   call {call}: FAILED  {run.error}")
                    break
                run.calls.append(result)
                print(
                    f"   call {result.call}:  prompt={result.prompt_tokens:>6}  "
                    f"cached={result.cached_tokens:>6}  "
                    f"write={result.cache_write_tokens:>6}  ->  {result.answer[:60]}"
                )
            print()

    header = (
        f"{'provider':<10} {'model':<38} {'prompt':>7} {'cached':>7} {'hit%':>6} {'distinct':>9}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        if not run.calls:
            print(f"{run.provider:<10} {run.model:<38} {'-':>7} {'-':>7} {'-':>6} {'ERROR':>9}")
            continue
        last = run.calls[-1]
        distinct = len({c.answer for c in run.calls})
        hit = 100 * last.cached_tokens / last.prompt_tokens if last.prompt_tokens else 0
        print(
            f"{run.provider:<10} {run.model:<38} {last.prompt_tokens:>7} "
            f"{last.cached_tokens:>7} {hit:>5.0f}% {distinct:>4}/{len(run.calls):<4}"
        )

    with open(args.out, "w") as fh:
        json.dump(
            {
                "reps": args.reps,
                "calls_per_model": args.calls,
                "context_chars": len(context),
                "question_name": args.question,
                "question": question,
                "runs": [asdict(r) for r in runs],
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {args.out}")

    warmed = [r for r in runs if len(r.calls) > 1]
    if warmed and all(r.calls[-1].cached_tokens == 0 for r in warmed):
        print("\nNo hits anywhere — prefix is probably too short. Raise --reps.")


if __name__ == "__main__":
    main()
