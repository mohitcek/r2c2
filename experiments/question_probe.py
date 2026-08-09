"""Find questions the models can't answer consistently.

The control refund question is unambiguous, so every sample agrees and
noncontradiction pins at ~0.999 — a fine caching demo, a useless uncertainty demo.
This tries candidates against the same policy looking for one where the model
contradicts itself. The productive ones aren't hard questions; they're questions the
policy doesn't answer, where the model invents a rule and invents a different one next
time.

Uses a one-clause policy rather than the full prefix — this is about the question, not
about caching.

    python experiments/question_probe.py
    python experiments/question_probe.py --models gpt-5.6-terra --questions airline-fault
"""

import argparse
import json

from policy import POLICY_CLAUSE

from r2c2.providers import run_google, run_openai
from r2c2.scoring import consistency_scores_batch

POLICY = POLICY_CLAUSE.format(i=1)

# (name, what gap it probes, question)
CANDIDATES = [
    (
        "control",
        "policy answers it outright",
        "A customer booked a non-refundable fare 20 hours ago and wants to cancel. "
        "Are they eligible for a refund? Answer in one sentence.",
    ),
    (
        "boundary",
        "'within 24 hours' — inclusive or not?",
        "A customer booked a non-refundable fare exactly 24 hours ago, to the minute, "
        "and wants to cancel. Are they eligible for a refund? Answer in one sentence.",
    ),
    (
        "after-departure",
        "policy covers 'before departure', silent after",
        "A customer bought a fully refundable fare and has already completed the flight. "
        "They now want a refund. Are they eligible? Answer in one sentence.",
    ),
    (
        "bereavement-gap",
        "retroactive ban applies only after travel; before travel is unwritten",
        "A customer booked a bereavement fare but never got pre-approval. They have not "
        "travelled yet. Can pre-approval still be granted for their booking? "
        "Answer in one sentence.",
    ),
    (
        "clause-conflict",
        "two clauses apply and point opposite ways",
        "A customer booked a bereavement fare that was also sold as fully refundable. "
        "They have completed travel and want a refund. Are they eligible? "
        "Answer in one sentence.",
    ),
    (
        "airline-fault",
        "policy silent on who cancelled",
        "A customer booked a non-refundable fare 30 hours ago. The airline then cancelled "
        "their flight outright. Are they eligible for a refund? Answer in one sentence.",
    ),
    (
        "business-days",
        "needs unstated assumptions to compute",
        "A refund is approved on Friday. Using the policy's 7-business-day rule, on which "
        "day does the money arrive? Answer in one sentence.",
    ),
    (
        "unwritten-policy",
        "asks for a rule the policy never states",
        "What is Northwind Air's refund policy for a non-refundable fare when the "
        "passenger misses the flight because of illness? Answer in one sentence.",
    ),
    (
        "weather",
        "force majeure — policy silent on weather",
        "A customer booked a non-refundable fare a month ago. A hurricane has now closed "
        "the destination airport and their flight cannot operate. Are they eligible for a "
        "refund? Answer in one sentence.",
    ),
    (
        "schedule-change",
        "airline moved the flight; policy silent on schedule changes",
        "A customer booked a non-refundable fare three weeks ago. The airline has now "
        "moved the departure time by 9 hours. The customer cannot make the new time and "
        "wants their money back. Are they eligible for a refund? Answer in one sentence.",
    ),
    (
        "duplicate-booking",
        "obvious mistake, but outside the 24h window",
        "A customer accidentally booked the same non-refundable flight twice, 30 hours "
        "ago, and wants a refund on the duplicate. Are they eligible? "
        "Answer in one sentence.",
    ),
    (
        "medical",
        "sympathy pull vs strict clause",
        "A customer with a non-refundable fare booked last month has been hospitalized "
        "and cannot fly. Are they eligible for a refund? Answer in one sentence.",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="gpt-5.6-terra,gemini-3.6-flash")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--questions", default="", help="comma-separated subset")
    parser.add_argument("--out", default="receipts/question_probe.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted = {q.strip() for q in args.questions.split(",") if q.strip()}
    unknown = wanted - {c[0] for c in CANDIDATES}
    if unknown:
        raise SystemExit(f"unknown question(s): {', '.join(sorted(unknown))}")
    candidates = [c for c in CANDIDATES if not wanted or c[0] in wanted]

    collected = {}
    for model in models:
        runner = run_google if model.startswith("gemini") else run_openai
        for name, _gap, question in candidates:
            answers = []
            for call in range(1, args.samples + 1):
                try:
                    answers.append(runner(model, POLICY, question, call).answer)
                except Exception as exc:
                    print(f"  {model} / {name}: FAILED {type(exc).__name__}: {exc}")
                    break
            if len(answers) < 2:
                continue
            collected[(model, name)] = answers
            print(f"  {model} / {name}: {len(answers)} samples, {len(set(answers))} distinct")

    if not collected:
        raise SystemExit("nothing collected")

    keys = list(collected)
    print(f"\nscoring {sum(len(collected[k]) for k in keys)} answers\n")
    all_scores = consistency_scores_batch([collected[k] for k in keys])

    gaps = {name: gap for name, gap, _ in CANDIDATES}
    rows = [
        {
            "model": model,
            "question": name,
            "probes": gaps[name],
            "noncontradiction_mean": round(sum(scores) / len(scores), 4),
            "noncontradiction_min": round(min(scores), 4),
            "distinct": len(set(collected[(model, name)])),
            "samples": len(scores),
            "answers": collected[(model, name)],
        }
        for (model, name), scores in zip(keys, all_scores, strict=True)
    ]
    rows.sort(key=lambda r: r["noncontradiction_mean"])

    print(f"{'model':<18} {'question':<18} {'mean':>7} {'min':>7} {'distinct':>9}")
    print("-" * 64)
    for r in rows:
        print(
            f"{r['model']:<18} {r['question']:<18} {r['noncontradiction_mean']:>7.3f} "
            f"{r['noncontradiction_min']:>7.3f} {r['distinct']:>5}/{r['samples']:<3}"
        )

    with open(args.out, "w") as fh:
        json.dump({"scorer": "noncontradiction", "results": rows}, fh, indent=2)
    print(f"\nwrote {args.out} — lowest score is the question worth talking about")


if __name__ == "__main__":
    main()
