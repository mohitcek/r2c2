"""Published per-token prices for the measured models.

Hardcoded and dated — re-verify against the provider pricing pages before quoting
these anywhere. Everything downstream (economics, CLI, experiments) reads prices
from here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_AS_OF = "2026-07-31"

MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """Dollars per 1M tokens.

    cached=None means the provider publishes no cached rate for this model, so cache
    hits bill at the full input price (Together does this per-model — a cache hit is
    an infrastructure fact, a discount is a pricing decision). cache_write=None means
    no separate write premium (Anthropic is the only measured provider with one).
    """

    input: float
    output: float
    cached: float | None = None
    cache_write: float | None = None

    @property
    def cached_ratio(self) -> float:
        """c — cached rate as a multiple of the input rate (1.0 if no cached price)."""
        return self.cached / self.input if self.cached is not None else 1.0

    @property
    def write_ratio(self) -> float:
        """w — cache-write rate as a multiple of the input rate (1.0 if no premium)."""
        return self.cache_write / self.input if self.cache_write is not None else 1.0

    @property
    def output_ratio(self) -> float:
        """k — output price as a multiple of the input price."""
        return self.output / self.input


PRICES: dict[str, ModelPrice] = {
    "gpt-5.6-terra": ModelPrice(input=2.50, output=15.00, cached=0.25),
    "claude-opus-5": ModelPrice(input=5.00, output=25.00, cached=0.50, cache_write=6.25),
    "gemini-3.6-flash": ModelPrice(input=1.50, output=7.50, cached=0.15),
    "deepseek-ai/DeepSeek-V4-Pro": ModelPrice(input=1.74, output=3.48, cached=0.20),
    "Qwen/Qwen3.6-Plus": ModelPrice(input=0.50, output=3.00),
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPrice(input=0.88, output=0.88),
}


def price_for(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError:
        known = ", ".join(sorted(PRICES))
        raise KeyError(f"no price on file for {model!r}; known models: {known}") from None
