"""Offline tests: the cost math against known values and the shipped receipts."""

import json
from pathlib import Path

import pytest

from r2c2 import (
    PRICES,
    ModelPrice,
    cost_floor,
    cost_multiplier,
    estimate,
    measured_costs,
    required_context,
)

RECEIPT = Path(__file__).parent / "data" / "receipt_uq.json"


def test_floor_openai_and_anthropic():
    # 90% cached discount, no write premium: 1 + 5 * 0.1
    assert cost_floor(6, PRICES["gpt-5.6-terra"]) == pytest.approx(1.50)
    # Anthropic's 1.25x cache write lifts the floor: 1.25 + 5 * 0.1
    assert cost_floor(6, PRICES["claude-opus-5"]) == pytest.approx(1.75)


def test_headline_multiplier():
    # The blog's ~21k-token receipt: 6 samples on gpt-5.6-terra cost ~1.56x one call.
    m = cost_multiplier(C=21_000, Q=50, O=40, n_samples=6, price=PRICES["gpt-5.6-terra"])
    assert m == pytest.approx(1.56, abs=0.01)


def test_no_cached_price_pins_at_n():
    # c = 1 collapses the formula to M = N regardless of how much caches.
    qwen = PRICES["Qwen/Qwen3.6-Plus"]
    for context in (1_000, 100_000, 10_000_000):
        assert cost_multiplier(context, 50, 40, 6, qwen) == pytest.approx(6.0)


def test_closed_form_matches_measured_costs_exactly():
    # When write tokens equal cached tokens, cost_multiplier() must agree with
    # measured_costs() to machine precision — the same model in two shapes.
    for price in PRICES.values():
        C, Q, O, n = 21_000, 50, 400, 6
        pair = measured_costs(C + Q, C, C, O, price, n)
        assert pair.scored / pair.unscored == pytest.approx(cost_multiplier(C, Q, O, n, price))


def test_closed_form_matches_receipt():
    # The measured runs behind the blog post (shipped as a fixture), reproduced
    # from token counts alone.
    receipt = json.loads(RECEIPT.read_text())
    n = receipt["calls_per_model"]
    checked = 0
    for run in receipt["runs"]:
        model, calls = run["model"], run["calls"]
        if model not in PRICES or not calls:
            continue
        price = PRICES[model]
        prompt = calls[0]["prompt_tokens"]
        cached = max(c["cached_tokens"] for c in calls)
        written = max(c["cache_write_tokens"] for c in calls)
        out_tokens = sum(c["output_tokens"] for c in calls) / len(calls)

        pair = measured_costs(prompt, cached, written, out_tokens, price, n)
        predicted = cost_multiplier(cached, prompt - cached, out_tokens, n, price)
        # Residual comes from cache-write vs cache-read counts differing slightly.
        assert pair.scored / pair.unscored == pytest.approx(predicted, abs=0.05), model
        checked += 1
    assert checked >= 4  # the receipt covers 6 models; most must be checkable


def test_estimate_consistency_and_threshold():
    est = estimate(context_tokens=21_000, output_tokens=40, model="gpt-5.6-terra")
    assert est.multiplier == pytest.approx(
        cost_multiplier(21_000, 50, 40, 6, PRICES["gpt-5.6-terra"])
    )
    assert est.cost_scored == pytest.approx(est.multiplier * est.cost_single)
    assert est.within(2.0) and not est.within(1.5)
    assert est.surcharge == pytest.approx(est.multiplier - 1)


def test_estimate_requires_exactly_one_price_source():
    with pytest.raises(ValueError):
        estimate(context_tokens=1_000, output_tokens=40)
    with pytest.raises(ValueError):
        estimate(
            context_tokens=1_000,
            output_tokens=40,
            model="gpt-5.6-terra",
            price=ModelPrice(input=1.0, output=2.0),
        )
    with pytest.raises(KeyError):
        estimate(context_tokens=1_000, output_tokens=40, model="no-such-model")


def test_required_context_inverts_the_multiplier():
    price = PRICES["gpt-5.6-terra"]
    c = required_context(output_tokens=40, model="gpt-5.6-terra", threshold=1.8)
    # exactly on the boundary, under with more context, over with less
    assert cost_multiplier(c, 50, 40, 6, price) == pytest.approx(1.8)
    assert cost_multiplier(c * 2, 50, 40, 6, price) < 1.8
    assert cost_multiplier(c / 2, 50, 40, 6, price) > 1.8
    # the blog's worked example: f/(1-f) = 14, Q + kO = 290
    assert c == pytest.approx(4_060)


def test_required_context_unachievable_at_or_below_floor():
    # thresholds at the floor are approached from above, never crossed
    assert required_context(output_tokens=40, model="gpt-5.6-terra", threshold=1.5) is None
    assert required_context(output_tokens=40, model="claude-opus-5", threshold=1.6) is None


def test_required_context_no_cached_price():
    # c = 1 pins the multiplier at N: nothing below 6x is reachable, 6x is free
    assert required_context(output_tokens=40, model="Qwen/Qwen3.6-Plus", threshold=5.99) is None
    assert required_context(output_tokens=40, model="Qwen/Qwen3.6-Plus", threshold=6.0) == 0.0


def test_required_context_scales_with_output_at_k():
    # dC/dO = f/(1-f) * k — at threshold 1.8 on terra that is 14 * 6 = 84 context
    # tokens per output token, the C >> kO rule made exact
    short = required_context(output_tokens=40, model="gpt-5.6-terra", threshold=1.8)
    long = required_context(output_tokens=1_040, model="gpt-5.6-terra", threshold=1.8)
    assert long - short == pytest.approx(84 * 1_000)


def test_required_context_requires_exactly_one_price_source():
    with pytest.raises(ValueError):
        required_context(output_tokens=40)
    with pytest.raises(ValueError):
        required_context(
            output_tokens=40,
            model="gpt-5.6-terra",
            price=ModelPrice(input=1.0, output=2.0),
        )


def test_summary_names_a_missing_cached_rate():
    """The reason a Together model sits at 6x should be on the line, not inferred."""
    est = estimate(context_tokens=21_000, output_tokens=40,
                   model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    assert est.cached_ratio == 1.0
    assert "no cached rate" in est.summary()
    assert "floor 6.00x" in est.summary()


def test_summary_shows_the_write_premium_that_lifts_the_floor():
    est = estimate(context_tokens=21_000, output_tokens=40, model="claude-opus-5")
    assert est.write_ratio == 1.25
    assert "w=1.25" in est.summary()
    assert "floor 1.75x" in est.summary()


def test_summary_omits_w_when_there_is_no_write_premium():
    est = estimate(context_tokens=21_000, output_tokens=40, model="gpt-5.6-terra")
    assert "c=0.10" in est.summary() and "w=" not in est.summary()
