"""The feasibility skill's script must keep agreeing with the package it wraps."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "skills" / "r2c2-feasibility" / "scripts" / "feasibility.py"


def run(*args):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )
    return proc.returncode, json.loads(proc.stdout)


def test_check_env_reports_prices_date_and_models():
    code, out = run("--check-env")
    assert code == 0 and out["ok"]
    assert out["prices_as_of"] and "gpt-5.6-terra" in out["models"]


def test_affordable_case_matches_package():
    from r2c2 import estimate

    code, out = run("--model", "gpt-5.6-terra", "--context", "21000", "--output", "40")
    assert code == 0 and out["verdict"] == "affordable"
    expected = estimate(context_tokens=21_000, output_tokens=40, model="gpt-5.6-terra")
    assert out["multiplier"] == pytest.approx(expected.multiplier, abs=1e-3)
    assert out["floor"] == pytest.approx(1.50)
    assert "Relative cost" in out["caveats"][0]


def test_not_yet_case_gives_the_crossover():
    code, out = run("--model", "gpt-5.6-terra", "--context", "2000", "--output", "40",
                    "--threshold", "1.8")
    assert code == 1 and out["verdict"] == "not_yet"
    assert out["required_context_tokens"] == pytest.approx(4060, abs=1)


def test_never_case_when_threshold_at_or_below_floor():
    code, out = run("--model", "claude-opus-5", "--context", "50000", "--output", "40",
                    "--threshold", "1.6")
    assert code == 1 and out["verdict"] == "never_at_this_threshold"
    assert out["required_context_tokens"] is None
    assert out["floor"] == pytest.approx(1.75)


def test_unknown_model_lists_known_ones():
    code, out = run("--model", "nope", "--context", "1000")
    assert code == 2 and not out["ok"]
    assert "claude-opus-5" in out["known_models"]


def test_sweep_is_monotone_and_reports_floor():
    code, out = run("--model", "claude-opus-5", "--output", "200",
                    "--sweep", "1000,5000,20000,100000")
    assert code == 0
    mults = [r["multiplier"] for r in out["sweep"]]
    assert mults == sorted(mults, reverse=True)          # falls as context grows
    assert all(m > out["floor"] for m in mults)           # never crosses the floor
    assert out["required_context_tokens"] is not None
