"""Shared fixture for the experiments: a synthetic airline policy and test questions.

The policy repeats one clause to build an arbitrarily long, deterministic prefix.
"control" is answered directly by the policy; "airline-fault" falls in a gap the
policy never covers, which is where models start contradicting themselves.
"""

POLICY_CLAUSE = (
    "Northwind Air refund policy, clause {i}: fully refundable fares may be "
    "refunded any time before departure. Non-refundable fares may be refunded "
    "only within 24 hours of booking. Refunds return to the original payment "
    "method within 7 business days. Bereavement fares require pre-approval "
    "and are never applied retroactively after travel is complete. "
)

QUESTIONS = {
    "control": (
        "A customer booked a non-refundable fare 20 hours ago and wants to cancel. "
        "Are they eligible for a refund? Answer in one sentence."
    ),
    "airline-fault": (
        "A customer booked a non-refundable fare 30 hours ago. The airline then "
        "cancelled their flight outright. Are they eligible for a refund? "
        "Answer in one sentence."
    ),
}

# Minimum cacheable prefix differs per provider (512 Anthropic / 1024 OpenAI /
# 4096 Gemini) and below it you get zeros with no error. 4096 isn't enough either:
# Gemini 3 Flash reports nothing between ~9k and ~17k tokens and only resumes above
# ~18k. 300 clauses (~21k tokens) is the smallest size that caches on all four.
DEFAULT_REPS = 300


def build_context(reps, nonce=""):
    # Nonce goes at the front. Caching matches from byte 0, so appending to a prefix
    # the provider has seen still hits; only changing the start forces a cold call.
    head = f"Policy revision {nonce}. " if nonce else ""
    return head + "".join(POLICY_CLAUSE.format(i=i) for i in range(reps))
