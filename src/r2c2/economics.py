"""What N-sample consistency scoring costs relative to one unscored call.

The folk objection to black-box UQ is "N samples = N x cost". Under prompt caching
the real number is

    M = N - f(N - w - (N-1)c)      f = C / (C + Q + kO)

where C is the cached context, Q the uncached input per call, O the output, and
c/w/k the cached/write/output rates as multiples of the input rate. f is the
cost-weighted cacheable share — output counts k times because it is priced k times
higher, which is why long answers hurt more than long contexts help.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .pricing import MILLION, ModelPrice, price_for


def cost_multiplier(C: float, Q: float, O: float, n_samples: int, price: ModelPrice) -> float:
    """Cost of N samples of the same prompt relative to one unscored call.

    Reproduces every measured model in tests/data/receipt_uq.json to within a few
    hundredths (the residual is cache-write vs cache-read token counts differing
    slightly on real calls).
    """
    f = C / (C + Q + price.output_ratio * O)
    return n_samples - f * (n_samples - price.write_ratio - (n_samples - 1) * price.cached_ratio)


def cost_floor(n_samples: int, price: ModelPrice) -> float:
    """Multiplier as C -> infinity. Not 1 + (N-1)c: the cache write counts too."""
    return price.write_ratio + (n_samples - 1) * price.cached_ratio


class CostPair(NamedTuple):
    """Dollars for one unscored call vs the N-sample scored request."""

    unscored: float
    scored: float


def measured_costs(
    prompt_tokens: float,
    cached_tokens: float,
    written_tokens: float,
    output_tokens: float,
    price: ModelPrice,
    n_samples: int,
) -> CostPair:
    """CostPair from measured token counts.

    The scored request is one cold call (pays the cache write) plus N-1 warm calls
    (read the cache); every call regenerates the output at full price.
    """
    cached_rate = price.cached if price.cached is not None else price.input
    write_rate = price.cache_write if price.cache_write is not None else price.input

    unscored = (prompt_tokens * price.input + output_tokens * price.output) / MILLION
    cold = (
        (prompt_tokens - written_tokens) * price.input
        + written_tokens * write_rate
        + output_tokens * price.output
    ) / MILLION
    warm = (
        (prompt_tokens - cached_tokens) * price.input
        + cached_tokens * cached_rate
        + output_tokens * price.output
    ) / MILLION
    return CostPair(unscored=unscored, scored=cold + (n_samples - 1) * warm)


@dataclass(frozen=True)
class Estimate:
    """The answer to "what would scoring this request actually cost me?"."""

    model: str | None
    n_samples: int
    context_tokens: float
    question_tokens: float
    output_tokens: float
    multiplier: float
    floor: float
    cost_single: float
    cost_scored: float
    cached_ratio: float
    write_ratio: float

    @property
    def surcharge(self) -> float:
        """Extra cost of scoring as a fraction of the unscored call (0.55 = +55%)."""
        return self.multiplier - 1.0

    def within(self, threshold: float = 2.0) -> bool:
        """True when the multiplier is at or below the caller's cost tolerance."""
        return self.multiplier <= threshold

    @property
    def rates(self) -> str:
        """The resolved rate ratios behind the floor — c=1 means nothing was discounted."""
        if self.cached_ratio >= 1.0:
            return "c=1.00 — no cached rate"
        note = f"c={self.cached_ratio:.2f}"
        if self.write_ratio != 1.0:
            note += f", w={self.write_ratio:.2f}"
        return note

    def summary(self) -> str:
        name = self.model or "custom price"
        return (
            f"{name}: {self.n_samples} samples cost {self.multiplier:.2f}x one call "
            f"(naive guess {self.n_samples}.00x, floor {self.floor:.2f}x at {self.rates}) — "
            f"${self.cost_single:.6f} -> ${self.cost_scored:.6f} per request"
        )


def required_context(
    output_tokens: float,
    *,
    model: str | None = None,
    price: ModelPrice | None = None,
    question_tokens: float = 50,
    n_samples: int = 6,
    threshold: float = 2.0,
) -> float | None:
    """Smallest context at which an N-sample check comes in at or under `threshold`.

    The multiplier is a straight line in the cacheable share f, so it inverts
    cleanly: f = (N - M) / (N - floor), then C = f/(1-f) * (Q + kO). This turns the
    gate question ("is scoring affordable here?") into a planning one ("from what
    context size does it become affordable?").

    Returns 0.0 when the threshold clears N — any context passes, scoring never
    costs more than N calls. Returns None when threshold <= floor: the floor is
    approached from above and never crossed, so no amount of context gets there —
    lower n_samples or find a provider with a cached rate instead.
    """
    if (model is None) == (price is None):
        raise ValueError("pass exactly one of model= or price=")
    if price is None:
        price = price_for(model)

    if threshold >= n_samples:
        return 0.0
    fl = cost_floor(n_samples, price)
    if threshold <= fl:
        return None
    f = (n_samples - threshold) / (n_samples - fl)
    return f / (1.0 - f) * (question_tokens + price.output_ratio * output_tokens)


def estimate(
    context_tokens: float,
    output_tokens: float,
    *,
    model: str | None = None,
    price: ModelPrice | None = None,
    question_tokens: float = 50,
    n_samples: int = 6,
) -> Estimate:
    """Price an N-sample consistency check before making any API call.

    Pass exactly one of `model` (looked up in PRICES) or `price`. Assumes the whole
    context caches, the cache stays warm across the N samples, and the samples are
    fired sequentially enough for call 1 to write the cache before the rest read it.
    """
    if (model is None) == (price is None):
        raise ValueError("pass exactly one of model= or price=")
    if price is None:
        price = price_for(model)

    m = cost_multiplier(context_tokens, question_tokens, output_tokens, n_samples, price)
    single = (
        (context_tokens + question_tokens) * price.input + output_tokens * price.output
    ) / MILLION
    return Estimate(
        model=model,
        n_samples=n_samples,
        context_tokens=context_tokens,
        question_tokens=question_tokens,
        output_tokens=output_tokens,
        multiplier=m,
        floor=cost_floor(n_samples, price),
        cost_single=single,
        cost_scored=m * single,
        cached_ratio=price.cached_ratio,
        write_ratio=price.write_ratio,
    )
