"""Offline tests for the check() gate — the skipped path makes no API calls."""

import pytest

from r2c2 import check


def test_check_skips_when_estimate_exceeds_threshold():
    # Tiny context + long expected answers: caching can't help, multiplier ~ N.
    # No API keys are set in CI, so this passing proves no network call happened.
    result = check(
        "gpt-5.6-terra",
        context="one short policy clause.",
        question="Is the customer eligible?",
        expected_output_tokens=2_000,
        threshold=2.0,
    )
    assert not result.sampled
    assert result.calls == ()
    assert result.confidence is None
    assert result.measured_multiplier is None
    assert result.estimate.multiplier > 2.0

    d = result.to_dict()
    assert d["sampled"] is False
    assert d["confidence"] is None
    assert d["answers"] == []


def test_check_unknown_model_fails_before_any_call():
    with pytest.raises(KeyError):
        check("no-such-model", context="x" * 1000, question="y?")
