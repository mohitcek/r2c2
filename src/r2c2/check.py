"""The whole loop in one call: estimate, and only if cheap enough, sample and score.

This is the function a routing layer, agent skill, or MCP tool wraps: given a
context and a question, work out what N samples would cost, skip if the surcharge
is unacceptable, otherwise collect the samples (call 1 warms the cache, calls 2-N
ride it) and score their agreement.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from .economics import Estimate, estimate, measured_costs
from .pricing import price_for
from .providers import CallResult, sample
from .scoring import consistency_scores

# Rough pre-call sizing; the measured multiplier uses real counts from the responses.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class CheckResult:
    model: str
    question: str
    estimate: Estimate
    sampled: bool  # False: the estimate exceeded the threshold, no API calls were made
    calls: tuple[CallResult, ...] = ()
    scores: tuple[float, ...] = field(default=(), repr=False)

    @property
    def answers(self) -> list[str]:
        return [c.answer for c in self.calls]

    @property
    def confidence(self) -> float | None:
        """Mean noncontradiction score, or None when the check was skipped."""
        return statistics.fmean(self.scores) if self.scores else None

    @property
    def distinct_answers(self) -> int:
        return len(set(self.answers))

    @property
    def measured_multiplier(self) -> float | None:
        """Scored/unscored cost ratio from the actual token counts, once sampled."""
        if not self.calls:
            return None
        price = price_for(self.model)
        pair = measured_costs(
            self.calls[0].prompt_tokens,
            max(c.cached_tokens for c in self.calls),
            max(c.cache_write_tokens for c in self.calls),
            statistics.fmean(c.output_tokens for c in self.calls),
            price,
            len(self.calls),
        )
        return pair.scored / pair.unscored

    def to_dict(self) -> dict:
        """JSON-friendly shape for CLIs and MCP tools."""
        return {
            "model": self.model,
            "question": self.question,
            "sampled": self.sampled,
            "estimated_multiplier": round(self.estimate.multiplier, 2),
            "measured_multiplier": (
                round(self.measured_multiplier, 2) if self.sampled else None
            ),
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "min_score": round(min(self.scores), 4) if self.scores else None,
            "distinct_answers": self.distinct_answers if self.sampled else None,
            "answers": self.answers,
        }


def check(
    model: str,
    context: str,
    question: str,
    *,
    n_samples: int = 6,
    threshold: float = 2.0,
    expected_output_tokens: float = 300,
    force: bool = False,
    on_call: Callable[[CallResult], None] | None = None,
) -> CheckResult:
    """Estimate -> sample -> score, gated on the estimated cost multiplier.

    Returns a CheckResult with sampled=False (and no API calls made) when the
    estimate exceeds `threshold`, unless force=True. A low `confidence` on the
    result means the model contradicts itself on this prompt — route it to a
    human, a retrieval retry, or a bigger model.
    """
    est = estimate(
        context_tokens=len(context) / CHARS_PER_TOKEN,
        output_tokens=expected_output_tokens,
        model=model,
        question_tokens=len(question) / CHARS_PER_TOKEN,
        n_samples=n_samples,
    )
    if not (est.within(threshold) or force):
        return CheckResult(model=model, question=question, estimate=est, sampled=False)

    calls = sample(model, context, question, n_samples, on_call=on_call)
    scores = consistency_scores([c.answer for c in calls])
    return CheckResult(
        model=model,
        question=question,
        estimate=est,
        sampled=True,
        calls=tuple(calls),
        scores=tuple(scores),
    )
