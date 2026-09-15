from __future__ import annotations

from typing import Any

import pytest

from gq import cli


class FakeDaemon:
    """Records requests and answers list_jobs from a fixed table of jobs."""

    def __init__(self, jobs: list[dict[str, Any]] | None = None) -> None:
        self.jobs = jobs or []
        self.requests: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any], **kwargs: object) -> dict[str, Any]:
        self.requests.append(payload)
        kind = payload["type"]
        if kind == "list_jobs":
            matched = [
                job
                for job in self.jobs
                if (payload.get("group") is None or job["group"] == payload["group"])
                and (not payload.get("states") or job["state"] in payload["states"])
                and (
                    not payload.get("id_ranges")
                    or any(low <= job["id"] <= high for low, high in payload["id_ranges"])
                )
            ]
            return {"ok": True, "jobs": matched, "held_groups": {}}
        if kind == "cancel_jobs":
            ids = payload["job_ids"]
            waiting = [j["id"] for j in self.jobs if j["id"] in ids and j["state"] == "WAITING"]
            return {
                "cancelled": waiting,
                "cancelling": [i for i in ids if i not in waiting],
                "skipped": [],
            }
        if kind == "cancel_job":
            return {"message": "cancellation requested"}
        if kind == "retry_jobs":
            return {"created": [], "skipped": [], "released_groups": []}
        if kind == "submit":
            return {"skipped": False, "job_id": 1, "state": "WAITING"}
        raise AssertionError(kind)

    def sent(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["type"] == kind]


def job(job_id: int, state: str, group: str = "sweep") -> dict[str, Any]:
    return {"id": job_id, "state": state, "group": group}


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch):
    def install(jobs: list[dict[str, Any]] | None = None, *, tty: bool = False) -> FakeDaemon:
        fake = FakeDaemon(jobs)
        monkeypatch.setattr(cli, "_daemon_request", fake)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: tty)
        return fake

    return install


# --- cancel --------------------------------------------------------------------------


def test_single_plain_id_keeps_the_original_cancel_request(daemon) -> None:
    fake = daemon()
    cli._cancel(["42"], None, None, assume_yes=False)
    assert fake.sent("cancel_job") == [{"type": "cancel_job", "job_id": 42}]
    assert fake.sent("cancel_jobs") == []


def test_cancel_refuses_to_guess_what_to_cancel(daemon) -> None:
    daemon()
    with pytest.raises(cli.CLIError, match="say what to cancel"):
        cli._cancel([], None, None, assume_yes=False)


def test_bulk_cancel_of_waiting_jobs_needs_no_confirmation(daemon) -> None:
    fake = daemon([job(1, "WAITING"), job(2, "WAITING"), job(3, "DONE")])
    cli._cancel([], "sweep", ["WAITING"], assume_yes=False)
    assert fake.sent("cancel_jobs")[0]["job_ids"] == [1, 2]


def test_killing_several_running_jobs_requires_confirmation(daemon) -> None:
    fake = daemon([job(1, "RUNNING"), job(2, "WAITING")], tty=False)
    with pytest.raises(cli.CLIError, match="--yes"):
        cli._cancel([], "sweep", None, assume_yes=False)
    assert fake.sent("cancel_jobs") == []


def test_declining_the_confirmation_cancels_nothing(daemon, monkeypatch) -> None:
    fake = daemon([job(1, "RUNNING"), job(2, "RUNNING")], tty=True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    cli._cancel(["1-2"], None, None, assume_yes=False)
    assert fake.sent("cancel_jobs") == []


def test_yes_cancels_exactly_the_listed_jobs(daemon, capsys) -> None:
    fake = daemon(
        [job(5, "RUNNING"), job(6, "WAITING"), job(7, "FAILED"), job(9, "RUNNING", "other")]
    )
    cli._cancel(["5-9"], "sweep", None, assume_yes=True)
    listed = fake.sent("list_jobs")[0]
    assert listed["id_ranges"] == [[5, 9]] and listed["group"] == "sweep"
    assert fake.sent("cancel_jobs")[0]["job_ids"] == [5, 6]
    assert "stopping 1 running job" in capsys.readouterr().out


# --- retry ---------------------------------------------------------------------------


def test_retry_by_group_only_sends_failed_or_cancelled_jobs(daemon) -> None:
    fake = daemon([job(1, "FAILED"), job(2, "DONE"), job(3, "CANCELLED"), job(4, "RUNNING")])
    cli._retry([], "sweep", None)
    assert fake.sent("list_jobs")[0]["states"] == ["FAILED", "CANCELLED"]
    assert fake.sent("retry_jobs")[0]["job_ids"] == [1, 3]


def test_retry_failed_only(daemon) -> None:
    fake = daemon([job(1, "FAILED"), job(3, "CANCELLED")])
    cli._retry([], "sweep", ["FAILED"])
    assert fake.sent("retry_jobs")[0]["job_ids"] == [1]


def test_named_jobs_are_sent_so_the_daemon_can_explain_refusals(daemon) -> None:
    """A DONE job or a typo'd id must be reported, not silently dropped."""
    fake = daemon([job(1, "DONE"), job(2, "FAILED")])
    cli._retry(["1", "2", "77"], None, None)
    assert "states" not in fake.sent("list_jobs")[0]
    assert fake.sent("retry_jobs")[0]["job_ids"] == [1, 2, 77]


def test_named_ids_outside_the_group_are_not_retried(daemon) -> None:
    fake = daemon([job(1, "FAILED", "sweep"), job(2, "FAILED", "other")])
    cli._retry(["1", "2"], "sweep", None)
    assert fake.sent("retry_jobs")[0]["job_ids"] == [1]


def test_retry_refuses_to_guess(daemon) -> None:
    daemon()
    with pytest.raises(cli.CLIError, match="say what to retry"):
        cli._retry([], None, None)


# --- submission ----------------------------------------------------------------------


def test_gq_group_environment_variable_applies_when_no_flag_is_given(daemon, monkeypatch) -> None:
    fake = daemon()
    monkeypatch.setenv("GQ_GROUP", "from-env")
    cli._submit(["true"], 1, None)
    cli._submit(["true"], 1, None, group="from-flag", key="k")
    first, second = fake.sent("submit")
    assert (first["group"], first["key"]) == ("from-env", None)
    assert (second["group"], second["key"]) == ("from-flag", "k")


def test_skipped_submission_is_reported_not_raised(daemon, monkeypatch, capsys) -> None:
    def skipped(payload: dict[str, Any], **kwargs: object) -> dict[str, Any]:
        return {
            "skipped": True,
            "job_id": 12,
            "state": "DONE",
            "message": "key 'k' is already DONE as job 12",
        }

    monkeypatch.setattr(cli, "_daemon_request", skipped)
    cli._submit(["true"], 1, None, key="k")
    assert capsys.readouterr().out.strip() == "Skipped: key 'k' is already DONE as job 12"
