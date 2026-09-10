from __future__ import annotations

from pathlib import Path

import pytest

from gq.cli import CLIError, _filter_env, _parse_duration, _script_directives


def test_script_directives(tmp_path: Path) -> None:
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/bash\n#gq --gpus=2\n#gq --name=training run\necho ok\n")
    assert _script_directives(script) == (2, "training run")


def test_unknown_script_directive_is_rejected(tmp_path: Path) -> None:
    script = tmp_path / "train.sh"
    script.write_text("#gq --priority=high\n")
    with pytest.raises(CLIError, match="unsupported"):
        _script_directives(script)


def test_secret_looking_variables_are_dropped_by_default() -> None:
    environ = {
        "PATH": "/usr/bin",
        "HF_TOKEN": "hf_xxx",
        "AWS_SECRET_ACCESS_KEY": "xxx",
        "OPENAI_API_KEY": "sk-xxx",
        "MY_PASSWORD": "hunter2",
        "SSH_KEY": "xxx",
        "CUDA_VISIBLE_DEVICES": "0",
        "MONKEY_BUSINESS": "fine",
    }
    kept, dropped = _filter_env(environ, keep_all=False)
    assert dropped == [
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "MY_PASSWORD",
        "OPENAI_API_KEY",
        "SSH_KEY",
    ]
    # Ordinary variables the job needs must survive, including near-miss names.
    assert kept == {
        "PATH": "/usr/bin",
        "CUDA_VISIBLE_DEVICES": "0",
        "MONKEY_BUSINESS": "fine",
    }


def test_env_all_and_env_keep_override_the_denylist() -> None:
    environ = {"PATH": "/usr/bin", "HF_TOKEN": "hf_xxx", "AWS_SECRET_ACCESS_KEY": "x"}
    kept, dropped = _filter_env(environ, keep_all=True)
    assert kept == environ and dropped == []

    kept, dropped = _filter_env(environ, keep_all=False, keep=["HF_TOKEN"])
    assert kept["HF_TOKEN"] == "hf_xxx"
    assert dropped == ["AWS_SECRET_ACCESS_KEY"]


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30s", 30), ("45m", 2700), ("12h", 43200), ("14d", 1209600), ("2w", 1209600)],
)
def test_parse_duration(text: str, seconds: int) -> None:
    assert _parse_duration(text).total_seconds() == seconds


@pytest.mark.parametrize("text", ["", "14", "d", "-1d", "14 d", "14y", "1.5h"])
def test_parse_duration_rejects_malformed_input(text: str) -> None:
    with pytest.raises(CLIError, match="invalid duration"):
        _parse_duration(text)
