from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gq.cli import (
    CLIError,
    ScriptDirectives,
    _filter_env,
    _parse_duration,
    _parse_selectors,
    _script_directives,
)


def test_script_directives(tmp_path: Path) -> None:
    script = tmp_path / "train.sh"
    script.write_text("#!/bin/bash\n#gq --gpus=2\n#gq --name=training run\necho ok\n")
    assert _script_directives(script) == ScriptDirectives(gpus=2, name="training run")


def test_group_and_key_directives(tmp_path: Path) -> None:
    script = tmp_path / "cell.sh"
    script.write_text("#gq --group=xsum-ablation\n#gq --key=xsum/kl/seed1\nrun\n")
    assert _script_directives(script) == ScriptDirectives(
        group="xsum-ablation", key="xsum/kl/seed1"
    )


def test_empty_key_directive_is_rejected(tmp_path: Path) -> None:
    script = tmp_path / "cell.sh"
    script.write_text("#gq --key=\n")
    with pytest.raises(CLIError, match="empty key"):
        _script_directives(script)


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


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["12"], [(12, 12)]),
        (["300-440"], [(300, 440)]),
        (["1", "5-7", "9"], [(1, 1), (5, 7), (9, 9)]),
    ],
)
def test_parse_selectors(tokens: list[str], expected: list[tuple[int, int]]) -> None:
    assert _parse_selectors(tokens) == expected


@pytest.mark.parametrize("token", ["", "abc", "12-", "-5", "10-5", "1.5", "1,2"])
def test_parse_selectors_rejects_malformed_input(token: str) -> None:
    with pytest.raises(CLIError):
        _parse_selectors([token])


def test_closed_stdout_exits_quietly() -> None:
    """`gq ps | head` must not print a BrokenPipeError when head stops reading."""
    code = (
        "from gq import cli\n"
        "cli._print_groups = lambda as_json: [print('x' * 80) for _ in range(50000)]\n"
        "raise SystemExit(cli.main(['groups']))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None and process.stderr is not None
    process.stdout.read(100)
    process.stdout.close()
    errors = process.stderr.read()
    assert process.wait(timeout=30) == 141
    assert b"BrokenPipe" not in errors and b"Exception ignored" not in errors
