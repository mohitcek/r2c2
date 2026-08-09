"""Score a receipt's answers for self-consistency, post-hoc — zero extra API calls.

    python experiments/confidence_score.py --receipt receipts/receipt_uq.json
"""

import argparse
import json

from r2c2.scoring import consistency_scores_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default="receipts/receipt.json")
    parser.add_argument("--out", default="receipts/confidence.json")
    args = parser.parse_args()

    with open(args.receipt) as fh:
        receipt = json.load(fh)

    groups = [
        ((run["provider"], run["model"]), [c["answer"] for c in run["calls"]])
        for run in receipt["runs"]
        if len(run["calls"]) >= 2
    ]
    if not groups:
        raise SystemExit("no runs with 2+ answers in the receipt")

    print(f"scoring {sum(len(a) for _, a in groups)} answers from {len(groups)} models\n")
    all_scores = consistency_scores_batch([answers for _, answers in groups])

    print(f"{'provider':<10} {'model':<38} {'mean':>7} {'min':>7} {'n':>4}")
    print("-" * 70)
    out = []
    for ((provider, model), _), scores in zip(groups, all_scores, strict=True):
        mean = sum(scores) / len(scores)
        out.append(
            {
                "provider": provider,
                "model": model,
                "noncontradiction_mean": round(mean, 4),
                "noncontradiction_min": round(min(scores), 4),
                "n": len(scores),
                "scores": [round(s, 4) for s in scores],
            }
        )
        print(f"{provider:<10} {model:<38} {mean:>7.3f} {min(scores):>7.3f} {len(scores):>4}")

    with open(args.out, "w") as fh:
        json.dump({"scorer": "noncontradiction", "models": out}, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("Scores rank consistency on this prompt. They are not calibrated "
          "probabilities, and consistency is not correctness.")


if __name__ == "__main__":
    main()
