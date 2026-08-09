"""CLI smoke tests for the offline subcommands."""

import json

from r2c2.cli import main


def test_models_lists_prices(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "gpt-5.6-terra" in out
    assert "claude-opus-5" in out


def test_estimate_below_threshold(capsys):
    code = main(
        [
            "estimate",
            "--model", "gpt-5.6-terra",
            "--context-tokens", "21000",
            "--output-tokens", "40",
            "--json",
        ]
    )
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert 1.5 < result["multiplier"] < 1.6


def test_estimate_exit_code_above_threshold(capsys):
    # Short prompt, long answers: caching can't help, so the check is "expensive".
    code = main(
        [
            "estimate",
            "--model", "gpt-5.6-terra",
            "--context-tokens", "500",
            "--output-tokens", "2000",
        ]
    )
    assert code == 1
    assert "expensive" in capsys.readouterr().out


def test_check_skip_path_is_offline(tmp_path, capsys):
    # The gate must refuse before touching any provider SDK or network.
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("one short policy clause.")
    code = main(
        [
            "check",
            "--model", "gpt-5.6-terra",
            "--context-file", str(ctx),
            "--question", "Is the customer eligible?",
            "--output-tokens", "2000",
            "--json",
        ]
    )
    assert code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["sampled"] is False
