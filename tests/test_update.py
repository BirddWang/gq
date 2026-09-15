from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from gq import cli
from gq import update as updater
from gq.update import Installation, classify_installation, is_newer


def classify(
    prefix: Path,
    *,
    installed: bool = True,
    direct_url: dict[str, Any] | None = None,
    has_pip: bool = True,
    on_path: tuple[str, ...] = ("uv", "pipx"),
) -> Installation:
    return classify_installation(
        prefix=prefix,
        executable=str(prefix / "bin" / "python"),
        installed=installed,
        direct_url=direct_url,
        has_pip=has_pip,
        which=lambda name: f"/usr/bin/{name}" if name in on_path else None,
    )


# --- install detection -------------------------------------------------------------


def test_uv_tool_installed_from_pypi_upgrades_with_uv(tmp_path: Path) -> None:
    (tmp_path / "uv-receipt.toml").write_text("")
    found = classify(tmp_path)
    assert found.kind == "uv-tool"
    assert found.command == ["/usr/bin/uv", "tool", "upgrade", "gq-local"]


def test_uv_tool_installed_from_a_checkout_is_not_silently_switched_to_pypi(
    tmp_path: Path,
) -> None:
    """The maintainer's own setup: `uv tool install .` from a clone."""
    (tmp_path / "uv-receipt.toml").write_text("")
    found = classify(tmp_path, direct_url={"url": "file:///src/gq", "dir_info": {}})
    assert found.kind == "source"
    assert found.command is None
    assert found.advice is not None
    assert "uv tool install --force gq-local" in found.advice


def test_editable_install_is_told_to_git_pull(tmp_path: Path) -> None:
    found = classify(tmp_path, direct_url={"url": "file:///src/gq", "dir_info": {"editable": True}})
    assert found.command is None
    assert found.advice is not None and "git pull" in found.advice


def test_vcs_install_is_not_upgraded_from_pypi(tmp_path: Path) -> None:
    found = classify(
        tmp_path,
        direct_url={"url": "https://github.com/BirddWang/gq", "vcs_info": {"vcs": "git"}},
    )
    assert found.kind == "source"
    assert found.command is None


def test_wheel_file_install_follows_its_environment_manager(tmp_path: Path) -> None:
    found = classify(tmp_path, direct_url={"url": "file:///x.whl", "archive_info": {}})
    assert found.kind == "pip"
    assert found.command is not None


def test_pipx_upgrades_with_pipx(tmp_path: Path) -> None:
    (tmp_path / "pipx_metadata.json").write_text("{}")
    found = classify(tmp_path)
    assert found.command == ["/usr/bin/pipx", "upgrade", "gq-local"]


@pytest.mark.parametrize(
    ("marker", "tool"), [("uv-receipt.toml", "uv"), ("pipx_metadata.json", "pipx")]
)
def test_missing_manager_executable_gives_advice_instead_of_guessing(
    tmp_path: Path, marker: str, tool: str
) -> None:
    (tmp_path / marker).write_text("")
    found = classify(tmp_path, on_path=())
    assert found.command is None
    assert found.advice is not None and tool in found.advice


def test_plain_virtualenv_uses_its_own_pip(tmp_path: Path) -> None:
    found = classify(tmp_path)
    assert found.command == [
        str(tmp_path / "bin" / "python"),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "gq-local",
    ]


def test_uv_created_virtualenv_without_pip_uses_uv_pip(tmp_path: Path) -> None:
    found = classify(tmp_path, has_pip=False)
    assert found.command is not None
    assert found.command[:3] == ["/usr/bin/uv", "pip", "install"]
    assert "--python" in found.command


def test_environment_with_no_installer_gives_advice(tmp_path: Path) -> None:
    found = classify(tmp_path, has_pip=False, on_path=())
    assert found.command is None


def test_uninstalled_source_tree_is_not_updatable(tmp_path: Path) -> None:
    found = classify(tmp_path, installed=False)
    assert found.command is None


# --- versions ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "current", "expected"),
    [
        ("0.2.0", "0.1.2", True),
        ("0.1.2", "0.1.2", False),
        ("0.1.1", "0.1.2", False),  # never offer a downgrade between releases
        ("0.10.0", "0.9.0", True),  # numeric, not lexicographic
        ("1.0.0", "1.0", False),
        ("0.2.0", "0.2.0.dev1", True),
    ],
)
def test_is_newer(candidate: str, current: str, expected: bool) -> None:
    assert is_newer(candidate, current) is expected


def test_latest_release_reads_pypi_json(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"info": {"version": "9.9.9"}}).encode()
    monkeypatch.setattr(updater.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    assert updater.latest_release() == "9.9.9"


def test_latest_release_reports_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(updater.urllib.request, "urlopen", refuse)
    with pytest.raises(updater.UpdateError, match="could not reach PyPI"):
        updater.latest_release()


# --- the update flow ---------------------------------------------------------------


class FakeDaemon:
    """Stands in for the daemon socket and records what the update flow asked of it."""

    def __init__(self, *, running: bool, active: list[int], can_pause: bool = True) -> None:
        self.running = running
        self.active = active
        self.can_pause = can_pause
        self.paused = False
        self.stopped = False
        self.started = False

    def request(self, payload: dict[str, Any], **kwargs: object) -> dict[str, Any]:
        kind = payload["type"]
        if kind == "pause_queue":
            if not self.can_pause:
                raise cli.CLIError("invalid request: unknown request type: 'pause_queue'")
            self.paused = True
            return {"ok": True, "active_job_ids": list(self.active)}
        if kind == "resume_queue":
            self.paused = False
            return {"ok": True}
        if kind == "list_jobs":
            return {"ok": True, "jobs": [{"id": i, "state": "RUNNING"} for i in self.active]}
        raise AssertionError(f"unexpected request {kind}")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    def build(
        *,
        running: bool = True,
        active: list[int] | None = None,
        can_pause: bool = True,
        latest: str = "0.2.0",
        installation: Installation | None = None,
    ) -> tuple[FakeDaemon, list[list[str]]]:
        daemon = FakeDaemon(running=running, active=active or [], can_pause=can_pause)
        installer_calls: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> Any:
            installer_calls.append(command)
            return type("Completed", (), {"returncode": 0})()

        def stop(paths: object) -> None:
            daemon.stopped = True

        def start(paths: object, **kwargs: object) -> dict[str, Any]:
            daemon.started = True
            return {"pid": 42, "version": "0.2.0"}

        monkeypatch.setattr(cli, "__version__", "0.1.2")
        monkeypatch.setattr(updater, "latest_release", lambda: latest)
        monkeypatch.setattr(
            updater,
            "detect_installation",
            lambda: (
                installation
                or Installation("uv-tool", "a uv tool environment", ["uv", "tool", "upgrade"])
            ),
        )
        monkeypatch.setattr(cli, "_ping", lambda paths: {"pid": 1} if daemon.running else None)
        monkeypatch.setattr(cli, "_daemon_request", daemon.request)
        monkeypatch.setattr(cli, "_stop_daemon", stop)
        monkeypatch.setattr(cli, "_start_daemon", start)
        monkeypatch.setattr(cli, "_installed_version", lambda: "0.2.0")
        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        return daemon, installer_calls

    return build


def test_up_to_date_changes_nothing(harness) -> None:
    daemon, installer = harness(latest="0.1.2")
    assert cli._update(check_only=False, wait=False, force=False) == 0
    assert installer == [] and not daemon.stopped and not daemon.paused


def test_check_only_never_installs(harness) -> None:
    daemon, installer = harness()
    assert cli._update(check_only=True, wait=False, force=False) == 0
    assert installer == [] and not daemon.stopped


def test_running_jobs_block_the_update_and_leave_the_queue_running(harness) -> None:
    daemon, installer = harness(active=[213, 214])
    with pytest.raises(cli.CLIError, match=r"(?s)213, 214 are running.*--wait.*--force"):
        cli._update(check_only=False, wait=False, force=False)
    assert installer == []
    assert not daemon.stopped
    assert not daemon.paused, "a refused update must not leave the queue paused"


def test_queue_is_paused_before_the_daemon_stops(harness) -> None:
    """Without the pause, a queued job could start between the check and the restart."""
    daemon, installer = harness(active=[])
    assert cli._update(check_only=False, wait=False, force=False) == 0
    assert daemon.paused and daemon.stopped and daemon.started
    assert installer == [["uv", "tool", "upgrade"]]


def test_force_updates_despite_running_jobs(harness, capsys) -> None:
    daemon, installer = harness(active=[7])
    assert cli._update(check_only=False, wait=False, force=True) == 0
    assert installer and daemon.stopped
    assert "recorded as FAILED" in capsys.readouterr().err


def test_wait_polls_until_running_jobs_finish(harness, monkeypatch) -> None:
    daemon, installer = harness(active=[7])
    polls = iter([[7], []])
    monkeypatch.setattr(cli, "_active_job_ids", lambda: next(polls))
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    assert cli._update(check_only=False, wait=True, force=False) == 0
    assert installer and daemon.stopped


def test_wait_refuses_against_a_daemon_that_cannot_pause(harness) -> None:
    daemon, installer = harness(active=[7], can_pause=False)
    with pytest.raises(cli.CLIError, match="predates queue pausing"):
        cli._update(check_only=False, wait=True, force=False)
    assert installer == [] and not daemon.stopped


def test_stopped_daemon_is_not_started_by_an_update(harness) -> None:
    daemon, installer = harness(running=False)
    assert cli._update(check_only=False, wait=False, force=False) == 0
    assert installer and not daemon.started


def test_source_installs_get_advice_not_an_installer(harness) -> None:
    daemon, installer = harness(
        installation=Installation("source", "installed from file:///src", None, "git pull")
    )
    with pytest.raises(cli.CLIError, match="git pull"):
        cli._update(check_only=False, wait=False, force=False)
    assert installer == [] and not daemon.stopped
