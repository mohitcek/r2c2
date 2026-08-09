"""r2c2 — Reuse Context, Recheck Consistency.

Ask the model again at cache prices, and score the agreement.

Consistency scoring samples the same prompt N times and checks the answers for
contradictions. The samples are byte-identical by construction — the best-case
workload for prompt caching — so on providers with a cached rate, N samples cost
nowhere near N times one call. This package prices that check before you run it
(`estimate`), tells you the context size where it becomes affordable
(`required_context`), runs the whole loop when it's cheap enough (`check`), and
exposes the pieces (`sample`, `consistency_scores`) for custom pipelines.

`estimate` is pure arithmetic with zero dependencies; sampling and scoring need
the `providers` / `scoring` extras respectively.
"""

from .check import CheckResult, check
from .economics import (
    CostPair,
    Estimate,
    cost_floor,
    cost_multiplier,
    estimate,
    measured_costs,
    required_context,
)
from .pricing import PRICES, PRICES_AS_OF, ModelPrice, price_for
from .providers import CallResult, sample
from .scoring import confidence, consistency_scores, consistency_scores_batch

__version__ = "0.1.0"

__all__ = [
    "PRICES",
    "PRICES_AS_OF",
    "CallResult",
    "CheckResult",
    "CostPair",
    "Estimate",
    "ModelPrice",
    "check",
    "confidence",
    "consistency_scores",
    "consistency_scores_batch",
    "cost_floor",
    "cost_multiplier",
    "estimate",
    "measured_costs",
    "price_for",
    "required_context",
    "sample",
]
