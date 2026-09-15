from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from . import update as updater
from .models import MANUAL_HOLD_REASON, JobState
from .paths import Paths
from .protocol import PROTOCOL_VERSION, ProtocolError, request


class CLIError(RuntimeError):
    pass


def _daemon_request(
    payload: dict[str, Any], *, auto_start: bool = True, timeout: float = 5.0
) -> dict[str, Any]:
    paths = Paths.from_environment()
    payload = {**payload, "protocol": PROTOCOL_VERSION}
    try:
        response = request(paths.socket, payload, timeout=timeout)
    except ProtocolError:
        if not auto_start:
            raise
        _start_daemon(paths, quiet=True)
        response = request(paths.socket, payload, timeout=timeout)
    if not response.get("ok"):
        raise CLIError(str(response.get("error", "daemon request failed")))
    return response


def _ping(paths: Paths) -> dict[str, Any] | None:
    try:
        response = request(paths.socket, {"type": "ping"}, timeout=0.5)
    except ProtocolError:
        return None
    return response if response.get("ok") else None


def _start_daemon(paths: Paths, *, quiet: bool = False) -> dict[str, Any]:
    paths.ensure()
    running = _ping(paths)
    if running:
        if not quiet:
            print(f"gq daemon is already running (pid {running['pid']})")
        return running
    log_handle = paths.daemon_log.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "gq.daemon"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        running = _ping(paths)
        if running:
            if not quiet:
                print(f"gq daemon started (pid {running['pid']})")
            return running
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise CLIError(f"daemon did not start; inspect {paths.daemon_log} for the startup error")


def _stop_daemon(paths: Paths) -> None:
    running = _ping(paths)
    if not running:
        print("gq daemon is not running")
        return
    response = request(paths.socket, {"type": "shutdown"}, timeout=2.0)
    if not response.get("ok"):
        raise CLIError(str(response.get("error", "daemon shutdown failed")))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _ping(paths):
        time.sleep(0.1)
    print(str(response.get("message", "daemon stopping")))


# Variables whose names suggest a credential. The submission environment is
# persisted verbatim into SQLite so queued jobs survive a daemon restart, which makes
# every captured secret a durable one.
SECRET_ENV_RE = re.compile(
    r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_KEY$|APIKEY|API_KEY|SESSION_KEY",
    re.IGNORECASE,
)
SECRET_ENV_PREFIXES = ("AWS_", "AZURE_", "GCP_", "GOOGLE_APPLICATION_")


def _filter_env(
    environ: dict[str, str], *, keep_all: bool, keep: Iterable[str] = ()
) -> tuple[dict[str, str], list[str]]:
    """Split the environment into what gets persisted and what gets dropped."""
    if keep_all:
        return dict(environ), []
    keep_set = set(keep)
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in environ.items():
        secret = SECRET_ENV_RE.search(key) or key.startswith(SECRET_ENV_PREFIXES)
        if secret and key not in keep_set:
            dropped.append(key)
        else:
            kept[key] = value
    return kept, sorted(dropped)


def _submit(
    argv: list[str],
    gpus: int,
    name: str | None,
    *,
    env_all: bool = False,
    env_keep: Iterable[str] = (),
    group: str | None = None,
    key: str | None = None,
) -> None:
    if not argv:
        raise CLIError("a command is required")
    if group is None:
        # Lets a sweep script set its group once instead of on every submission.
        group = os.environ.get("GQ_GROUP") or None
    if argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise CLIError("a command is required after --")
    env, dropped = _filter_env(
        dict(os.environ),
        keep_all=env_all or os.environ.get("GQ_ENV_ALL") == "1",
        keep=env_keep,
    )
    if dropped:
        # Say so rather than dropping silently: a job that needs HF_TOKEN would
        # otherwise fail much later, in a way indistinguishable from a bad run.
        print(
            f"gq: not persisting {len(dropped)} secret-looking variable(s): "
            f"{', '.join(dropped)}\n"
            "    use --env-all (or GQ_ENV_ALL=1) to keep them, or --env-keep NAME",
            file=sys.stderr,
        )
    response = _daemon_request(
        {
            "type": "submit",
            "argv": argv,
            "cwd": str(Path.cwd()),
            # Persisted so WAITING jobs survive daemon restarts. Never returned by
            # the normal IPC API.
            "env": env,
            "requested_gpus": gpus,
            "name": name,
            "group": group,
            "key": key,
        }
    )
    if response.get("skipped"):
        print(f"Skipped: {response['message']}")
        return
    print(f"Submitted job {response['job_id']}")


DIRECTIVE_RE = re.compile(r"^\s*#gq\s+(--(?:gpus|name|group|key))=(.*)\s*$")


@dataclass(frozen=True)
class ScriptDirectives:
    gpus: int | None = None
    name: str | None = None
    group: str | None = None
    key: str | None = None


def _script_directives(path: Path) -> ScriptDirectives:
    try:
        text = path.read_text()
    except (OSError, UnicodeError) as exc:
        raise CLIError(f"cannot read script {path}: {exc}") from exc
    gpus: int | None = None
    labels: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("#gq"):
            continue
        match = DIRECTIVE_RE.match(line)
        if not match:
            raise CLIError(f"unsupported or malformed #gq directive on line {line_number}")
        option, value = match.groups()
        value = value.strip()
        if option == "--gpus":
            try:
                gpus = int(value)
            except ValueError as exc:
                raise CLIError(f"invalid GPU count on line {line_number}: {value!r}") from exc
        else:
            if not value:
                raise CLIError(f"empty {option[2:]} on line {line_number}")
            labels[option[2:]] = value
    return ScriptDirectives(gpus, labels.get("name"), labels.get("group"), labels.get("key"))


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    lines.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rendered
    )
    return "\n".join(lines)


def _command_text(argv: list[str], maximum: int = 70) -> str:
    value = shlex.join(argv)
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


_ID_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


def _parse_selectors(tokens: Sequence[str]) -> list[tuple[int, int]]:
    """Turn `12` and `300-440` into inclusive id ranges."""
    ranges: list[tuple[int, int]] = []
    for token in tokens:
        match = _ID_RE.match(token.strip())
        if not match:
            raise CLIError(f"invalid job id or range {token!r}; use a number like 12 or 300-440")
        low = int(match.group(1))
        high = int(match.group(2) or low)
        if low > high:
            raise CLIError(f"invalid range {token!r}: the first id must not exceed the second")
        ranges.append((low, high))
    return ranges


def _list_request(
    tokens: Sequence[str] = (),
    group: str | None = None,
    states: Sequence[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "list_jobs", "limit": limit}
    if tokens:
        payload["id_ranges"] = [list(pair) for pair in _parse_selectors(tokens)]
    if group is not None:
        payload["group"] = group
    if states:
        payload["states"] = list(states)
    return _daemon_request(payload)


def _truncate(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


def _notice(message: str) -> None:
    # Flush the table first so a notice never lands above it when output is captured.
    sys.stdout.flush()
    print(message, file=sys.stderr)


def _print_holds(held: dict[str, str], groups: Iterable[str]) -> None:
    for group in sorted(set(groups) & set(held)):
        if held[group] == MANUAL_HOLD_REASON:
            hint = f"release it with 'gq daemon resume --group {group}'"
        else:
            hint = (
                f"fix the cause, then 'gq retry --group {group} --failed', or release it "
                f"as is with 'gq daemon resume --group {group}'"
            )
        _notice(f"gq: group {group} is held: {held[group]}\n    {hint}")


def _print_jobs(
    limit: int | None,
    as_json: bool = False,
    *,
    tokens: Sequence[str] = (),
    group: str | None = None,
    states: Sequence[str] | None = None,
) -> None:
    response = _list_request(tokens, group, states, limit)
    if as_json:
        print(json.dumps(response["jobs"], indent=2))
        return
    jobs = response["jobs"]
    held: dict[str, str] = response.get("held_groups") or {}
    show_group = any(job.get("group") for job in jobs)
    rows = []
    displayed: dict[int, str] = {}
    for job in jobs:
        gpus = ",".join(map(str, job["gpu_indices"])) or "-"
        state = job["state"]
        if state == JobState.WAITING.value and job.get("group") in held:
            state = "HELD"  # display only: the stored state is still WAITING
        displayed[job["id"]] = state
        row = [job["id"], state, gpus, job["pid"] or "-"]
        if show_group:
            row.append(_truncate(job.get("group") or "-", 24))
        row += [job["name"] or "-", _command_text(job["argv"])]
        rows.append(row)
    headers = ["JOB", "STATE", "GPUs", "PID"] + (["GROUP"] if show_group else [])
    print(_table([*headers, "NAME", "COMMAND"], rows))
    if group is not None and not tokens and not states and limit is None:
        superseded = _superseded_ids(jobs)
        counts: dict[str, int] = {}
        for job in jobs:
            if job["id"] not in superseded:
                counts[displayed[job["id"]]] = counts.get(displayed[job["id"]], 0) + 1
        summary = ", ".join(f"{n} {state}" for state, n in sorted(counts.items())) or "no jobs"
        if superseded:
            summary += f" (plus {len(superseded)} earlier attempts, since retried)"
        print(f"{group}: {summary}")
    shown_groups = {job["group"] for job in jobs if job.get("group")}
    if group is not None:
        shown_groups.add(group)  # a held group may have nothing left to list
    _print_holds(held, shown_groups)
    if response.get("queue_paused"):
        _notice("gq: the queue is paused; resume it with 'gq daemon resume'")


def _superseded_ids(jobs: Sequence[dict[str, Any]]) -> set[int]:
    """Ids of attempts that a later retry in the same lineage replaced."""
    latest: dict[int, int] = {}
    for job in jobs:
        root = job.get("retry_of") or job["id"]
        latest[root] = max(latest.get(root, 0), job["id"])
    return {job["id"] for job in jobs if latest[job.get("retry_of") or job["id"]] != job["id"]}


def _print_groups(as_json: bool) -> None:
    groups = _daemon_request({"type": "list_groups"})["groups"]
    if as_json:
        print(json.dumps(groups, indent=2))
        return
    if not groups:
        print("No groups yet. Submit with --group NAME, or set GQ_GROUP in a sweep script.")
        return
    active = (JobState.STARTING.value, JobState.RUNNING.value, JobState.CANCELLING.value)
    show_orphaned = any(g["counts"].get(JobState.ORPHANED.value) for g in groups)
    show_retried = any(g.get("retried") for g in groups)
    rows = []
    for entry in groups:
        counts = entry["counts"]
        row = [
            _truncate(entry["group"], 32),
            counts.get(JobState.WAITING.value, 0),
            sum(counts.get(state, 0) for state in active),
            counts.get(JobState.DONE.value, 0),
            counts.get(JobState.FAILED.value, 0),
            counts.get(JobState.CANCELLED.value, 0),
        ]
        if show_orphaned:
            row.append(counts.get(JobState.ORPHANED.value, 0))
        if show_retried:
            row.append(entry.get("retried", 0))
        row.append("held" if entry["held"] else "-")
        rows.append(row)
    headers = ["GROUP", "WAITING", "RUNNING", "DONE", "FAILED", "CANCELLED"]
    headers += ["ORPHANED"] if show_orphaned else []
    headers += ["RETRIED"] if show_retried else []
    print(_table([*headers, "STATUS"], rows))
    _print_holds(
        {entry["group"]: entry["hold_reason"] for entry in groups if entry["held"]},
        (entry["group"] for entry in groups),
    )
    if show_retried:
        _notice("counts cover each job's latest attempt; RETRIED is how many were replaced")


def _confirm(question: str, assume_yes: bool, flag: str = "--yes") -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        raise CLIError(f"this needs a terminal to confirm; pass {flag} to run it non-interactively")
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _cancel(
    tokens: Sequence[str],
    group: str | None,
    states: Sequence[str] | None,
    assume_yes: bool,
) -> None:
    if len(tokens) == 1 and tokens[0].isdigit() and group is None and not states:
        # One plain id keeps its original, stricter behavior: cancelling a job that
        # already finished is reported as an error rather than silently skipped.
        response = _daemon_request({"type": "cancel_job", "job_id": int(tokens[0])})
        print(f"Job {tokens[0]}: {response['message']}")
        return
    if not tokens and group is None and not states:
        raise CLIError("say what to cancel: job ids or ranges (12 300-440), --group, or --waiting")
    wanted = states or [JobState.WAITING.value, JobState.STARTING.value, JobState.RUNNING.value]
    jobs = _list_request(tokens, group, wanted)["jobs"]
    if not jobs:
        print("No matching jobs to cancel.")
        return
    running = [job["id"] for job in jobs if job["state"] != JobState.WAITING.value]
    waiting = len(jobs) - len(running)
    if running and len(jobs) > 1:
        question = (
            f"Cancel {_plural(len(jobs), 'job')} "
            f"({len(running)} running, {waiting} waiting)? Running jobs will be killed."
        )
        if not _confirm(question, assume_yes):
            print("Nothing was cancelled.")
            return
    # Cancel exactly what was listed (and confirmed), not whatever matches by now.
    ids = sorted(job["id"] for job in jobs)
    result = _daemon_request({"type": "cancel_jobs", "job_ids": ids}, timeout=30.0)
    parts = []
    if result["cancelled"]:
        parts.append(f"cancelled {_plural(len(result['cancelled']), 'waiting job')}")
    if result["cancelling"]:
        parts.append(f"stopping {_plural(len(result['cancelling']), 'running job')}")
    if result["skipped"]:
        parts.append(f"{len(result['skipped'])} had already finished")
    print((", ".join(parts) or "nothing to do").capitalize() + ".")


def _retry(tokens: Sequence[str], group: str | None, states: Sequence[str] | None) -> None:
    if not tokens and group is None:
        raise CLIError("say what to retry: job ids or ranges (216-218), or --group NAME")
    if tokens and not states:
        # Without a state filter, every matching job is sent so the daemon can say why
        # a DONE or RUNNING one was not retried, instead of it silently vanishing.
        found = {job["id"] for job in _list_request(tokens, group)["jobs"]}
        ids = sorted(found)
        if group is None:
            # A plain id that matched nothing is sent too, to be reported as missing.
            # Ranges are exempt: gaps in a range are normal after `gq rm`.
            named = {int(token) for token in tokens if token.strip().isdigit()}
            ids = sorted(found | named)
    else:
        wanted = states or [JobState.FAILED.value, JobState.CANCELLED.value]
        ids = sorted(job["id"] for job in _list_request(tokens, group, wanted)["jobs"])
    if not ids:
        print("No failed or cancelled jobs match.")
        return
    result = _daemon_request({"type": "retry_jobs", "job_ids": ids}, timeout=30.0)
    created = result["created"]
    if created:
        pairs = [f"{item['from']}->{item['job_id']}" for item in created]
        shown = ", ".join(pairs[:12]) + (f", and {len(pairs) - 12} more" if len(pairs) > 12 else "")
        print(f"Retried {_plural(len(created), 'job')}: {shown}")
    for group_name in result["released_groups"]:
        print(f"Released the hold on group {group_name}.")
    skipped = result["skipped"]
    if skipped:
        print(f"Skipped {_plural(len(skipped), 'job')}:")
        for item in skipped[:12]:
            print(f"  {item['id']}: {item['reason']}")
        if len(skipped) > 12:
            print(f"  ... and {len(skipped) - 12} more")


def _mib(value: int | None) -> str:
    return "-" if value is None else f"{value // (1024 * 1024)} MiB"


def _print_gpus(as_json: bool = False) -> None:
    response = _daemon_request({"type": "gpu_status"})
    if as_json:
        print(json.dumps(response["gpus"], indent=2))
        return
    rows = []
    for gpu in response["gpus"]:
        processes = gpu["processes"]
        if gpu["owner_job_id"] is not None:
            owner = f"job {gpu['owner_job_id']}"
        elif processes:
            owner = ",".join(
                (proc.get("name") or "external").rsplit("/", 1)[-1] for proc in processes
            )
        else:
            owner = "-"
        pids = ",".join(str(proc["pid"]) for proc in processes) or "-"
        uuid = gpu["uuid"] if len(gpu["uuid"]) <= 20 else gpu["uuid"][:17] + "…"
        rows.append(
            (
                gpu["index"],
                uuid,
                gpu["state"],
                owner,
                pids,
                _mib(gpu["used_memory"]),
            )
        )
    print(_table(["GPU", "UUID", "STATE", "OWNER", "PID", "MEMORY"], rows))


def _show(job_id: int, as_json: bool = False) -> dict[str, Any]:
    response = _daemon_request({"type": "show_job", "job_id": job_id})
    job: dict[str, Any] = response["job"]
    if as_json:
        print(json.dumps(job, indent=2))
        return job
    fields = [
        ("Job ID", job["id"]),
        ("Name", job["name"] or "-"),
        ("State", job["state"]),
        ("Submitted", job["submit_time"]),
        ("Started", job["start_time"] or "-"),
        ("Ended", job["end_time"] or "-"),
        ("Command", shlex.join(job["argv"])),
        ("Working dir", job["cwd"]),
        ("GPUs", ",".join(map(str, job["gpu_indices"])) or "-"),
        ("GPU UUIDs", ",".join(job["gpu_uuids"]) or "-"),
        ("PID", job["pid"] or "-"),
        ("PGID", job["pgid"] or "-"),
        ("Exit code", job["exit_code"] if job["exit_code"] is not None else "-"),
        ("Log", job["log_path"] or "-"),
        ("Failure", job["failure_reason"] or "-"),
    ]
    width = max(len(label) for label, _ in fields)
    print("\n".join(f"{label + ':':<{width + 2}}{value}" for label, value in fields))
    return job


def _logs(job_id: int, follow: bool) -> None:
    response = _daemon_request({"type": "show_job", "job_id": job_id})
    job = response["job"]
    path = Path(job["log_path"])
    if not follow:
        if path.exists():
            try:
                sys.stdout.buffer.write(path.read_bytes())
                sys.stdout.buffer.flush()
            except OSError as exc:
                raise CLIError(f"cannot read {path}: {exc}") from exc
        return

    offset = 0
    terminal = {state.value for state in JobState if state.terminal}
    try:
        while True:
            if path.exists():
                try:
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                    if chunk:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()
                        offset += len(chunk)
                except OSError as exc:
                    raise CLIError(f"cannot read {path}: {exc}") from exc
            response = _daemon_request({"type": "show_job", "job_id": job_id})
            state = response["job"]["state"]
            if state in terminal:
                # One more read catches bytes flushed immediately before exit.
                time.sleep(0.1)
                if path.exists() and path.stat().st_size > offset:
                    continue
                return
            time.sleep(0.25)
    except KeyboardInterrupt:
        return


DURATION_RE = re.compile(r"^(\d+)([smhdw])$")
DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_duration(text: str) -> timedelta:
    match = DURATION_RE.match(text.strip())
    if not match:
        raise CLIError(f"invalid duration {text!r}; use a count and a unit, such as 30d, 12h, 45m")
    amount, unit = match.groups()
    return timedelta(seconds=int(amount) * DURATION_UNITS[unit])


def _remove_jobs(job_ids: list[int]) -> None:
    response = _daemon_request({"type": "delete_jobs", "job_ids": job_ids})
    removed = response["removed"]
    print(f"Removed {len(removed)} job(s): {', '.join(map(str, removed))}")


def _clean_jobs(older_than: str, states: list[str] | None, assume_yes: bool) -> None:
    cutoff = (datetime.now(UTC) - _parse_duration(older_than)).isoformat()
    payload: dict[str, Any] = {"type": "clean_jobs", "cutoff": cutoff}
    if states:
        payload["states"] = states
    if not assume_yes:
        # Deleting history and logs is irreversible, so confirm unless told not to.
        if not sys.stdin.isatty():
            raise CLIError(
                "gq clean needs a terminal to confirm; pass --yes to run it non-interactively"
            )
        scope = f" in state {'/'.join(states)}" if states else ""
        answer = input(
            f"Delete all terminal jobs{scope} that ended more than {older_than} ago, "
            "including their logs? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Nothing was removed.")
            return
    response = _daemon_request(payload, timeout=30.0)
    removed = response["removed"]
    print(f"Removed {len(removed)} job(s) and their logs.")


ACTIVE_STATES = {JobState.STARTING.value, JobState.RUNNING.value, JobState.CANCELLING.value}


def _active_job_ids() -> list[int]:
    response = _daemon_request({"type": "list_jobs"}, auto_start=False)
    return [job["id"] for job in response["jobs"] if job["state"] in ACTIVE_STATES]


def _pause_queue() -> list[int] | None:
    """Pause the queue and return the jobs still running.

    Returns None when the daemon predates pausing. Pausing before looking at running
    jobs matters: otherwise a queued job can start between the check and the daemon
    stopping, and would then be recorded as FAILED by the restarted daemon.
    """
    try:
        response = _daemon_request({"type": "pause_queue"}, auto_start=False)
    except CLIError as exc:
        if "unknown request type" in str(exc):
            return None
        raise
    return [int(job_id) for job_id in response["active_job_ids"]]


def _resume_queue() -> None:
    _daemon_request({"type": "resume_queue"}, auto_start=False)


def _wait_for_running_jobs(active: list[int]) -> bool:
    """Wait, with the queue already paused, until no managed job is running.

    Returns False if the user interrupted, in which case the queue is resumed.
    """
    # Flushed explicitly: --wait can run for hours, often with output redirected.
    print("Queue paused: running jobs will finish, queued jobs will wait.", flush=True)
    print(
        "Waiting for running jobs to finish (Ctrl-C to cancel and resume the queue)...",
        flush=True,
    )
    reported: list[int] = []
    try:
        while active:
            if active != reported:
                print(f"  still running: {', '.join(map(str, active))}", flush=True)
                reported = active
            time.sleep(5)
            active = _active_job_ids()
    except KeyboardInterrupt:
        _resume_queue()
        print("\nUpdate cancelled; the queue is running again.")
        return False
    return True


def _installed_version() -> str | None:
    """Ask a fresh interpreter, since this process still has the old code loaded."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import gq; print(gq.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _jobs_are_running(job_ids: list[int]) -> str:
    listed = ", ".join(map(str, job_ids))
    return f"job {listed} is running" if len(job_ids) == 1 else f"jobs {listed} are running"


def _prepare_daemon_for_update(paths: Paths, *, wait: bool, force: bool) -> bool:
    """Get the daemon out of the way. Returns False if the user cancelled."""
    active = _pause_queue()
    can_pause = active is not None
    if active is None:
        active = _active_job_ids()
    try:
        if active and force:
            print(
                f"warning: updating while {_jobs_are_running(active)}; they will be "
                "recorded as FAILED when they finish",
                file=sys.stderr,
            )
        elif active and wait:
            if not can_pause:
                raise CLIError(
                    "the running daemon predates queue pausing, so --wait cannot stop a "
                    "queued job from starting mid-update. Wait until 'gq ps' shows nothing "
                    "running, or use --force."
                )
            if not _wait_for_running_jobs(active):
                return False
        elif active:
            raise CLIError(
                f"{_jobs_are_running(active).capitalize()}. Updating restarts the daemon, "
                "and a restarted daemon cannot collect the exit status of jobs it did not "
                "start, so they would be recorded as FAILED when they finish.\n"
                "    gq update --wait    pause the queue, let running jobs finish, then update\n"
                "    gq update --force   update now anyway"
            )
        _stop_daemon(paths)
    except CLIError:
        # Never leave a live daemon paused because the update did not go ahead.
        if can_pause and _ping(paths) is not None:
            _resume_queue()
        raise
    return True


def _update(check_only: bool, wait: bool, force: bool) -> int:
    installation = updater.detect_installation()
    try:
        latest = updater.latest_release()
    except updater.UpdateError as exc:
        raise CLIError(str(exc)) from exc
    print(f"Installed: gq {__version__} ({installation.description})")
    print(f"Latest:    gq {latest}")
    if not updater.is_newer(latest, __version__):
        print("gq is up to date.")
        return 0
    if check_only:
        print("An update is available; run 'gq update' to install it.")
        return 0
    if installation.command is None:
        raise CLIError(f"gq cannot update this installation itself: {installation.advice}")

    paths = Paths.from_environment()
    daemon_was_running = _ping(paths) is not None
    if daemon_was_running and not _prepare_daemon_for_update(paths, wait=wait, force=force):
        return 130

    print(f"Running: {shlex.join(installation.command)}", flush=True)
    restarted: dict[str, Any] | None = None
    try:
        result = subprocess.run(installation.command)
    except KeyboardInterrupt:
        print("\nInstaller interrupted.", file=sys.stderr)
        return 130
    finally:
        # Restart even if the installer failed or was interrupted: the queue should
        # keep moving on whichever version is now installed.
        if daemon_was_running:
            restarted = _start_daemon(paths, quiet=True)

    now = _installed_version()
    if result.returncode != 0:
        raise CLIError(
            f"the installer exited with status {result.returncode}; gq {now or 'unknown'} "
            "is still installed"
        )
    if now == __version__:
        print(
            f"warning: the installer succeeded but gq is still {__version__}; a version "
            "constraint may be pinning it",
            file=sys.stderr,
        )
    else:
        print(f"Updated gq {__version__} -> {now or 'unknown'}.")
    if restarted is not None:
        print(f"Daemon restarted (pid {restarted['pid']}, version {restarted['version']}).")
        if now is not None and restarted["version"] != now:
            print(
                f"warning: the daemon is running gq {restarted['version']}, not {now}; "
                "run 'gq daemon stop' and let the next command start it",
                file=sys.stderr,
            )
    return 0


def _add_env_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-all",
        action="store_true",
        help="persist secret-looking variables too (also GQ_ENV_ALL=1)",
    )
    parser.add_argument(
        "--env-keep",
        action="append",
        default=[],
        metavar="NAME",
        help="persist this variable even if it looks like a secret (repeatable)",
    )


def _add_label_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group",
        metavar="NAME",
        help="tag the job with a group, for bulk actions (default: $GQ_GROUP)",
    )
    parser.add_argument(
        "--key",
        metavar="KEY",
        help="skip the submission if a waiting, running, or DONE job already has this key",
    )


ALL_STATES = [state.value for state in JobState]


def _run_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="submit a GPU job")
    parser.add_argument("-g", "--gpus", type=int, required=True, help="number of GPUs")
    parser.add_argument("--name", help="job name")
    _add_label_options(parser)
    _add_env_options(parser)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gq", description="local NVIDIA GPU job scheduler")
    parser.add_argument("--version", action="version", version=f"gq {__version__}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    run = sub.add_parser("run", help="submit a command")
    run.add_argument("-g", "--gpus", type=int, required=True)
    run.add_argument("--name")
    _add_label_options(run)
    _add_env_options(run)
    run.add_argument("command", nargs=argparse.REMAINDER)

    submit = sub.add_parser("submit", help="submit a shell script")
    submit.add_argument("script", type=Path)
    submit.add_argument("-g", "--gpus", type=int)
    submit.add_argument("--name")
    _add_label_options(submit)
    _add_env_options(submit)

    ps = sub.add_parser("ps", help="list jobs")
    ps.add_argument("jobs", nargs="*", metavar="JOB", help="job ids or ranges such as 300-440")
    ps.add_argument("--group", metavar="NAME", help="only jobs in this group")
    ps.add_argument(
        "--state", action="append", choices=ALL_STATES, help="only jobs in this state (repeatable)"
    )
    ps.add_argument("--limit", type=int)
    ps.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    groups = sub.add_parser("groups", help="list job groups and their progress")
    groups.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    gpu = sub.add_parser("gpu", help="show physical and logical GPU state")
    gpu.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    show = sub.add_parser("show", help="show one job")
    show.add_argument("job_id", type=int)
    show.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    remove = sub.add_parser("rm", help="delete terminal jobs and their logs")
    remove.add_argument("job_id", type=int, nargs="+")

    clean = sub.add_parser("clean", help="delete old terminal jobs and their logs")
    clean.add_argument(
        "--older-than",
        default="30d",
        metavar="AGE",
        help="delete jobs that ended more than this long ago (default: 30d)",
    )
    clean.add_argument(
        "--state",
        action="append",
        choices=[state.value for state in JobState if state.terminal],
        help="restrict to these terminal states (repeatable)",
    )
    clean.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")

    logs = sub.add_parser("logs", help="read a job log")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("job_id", type=int)

    cancel = sub.add_parser("cancel", help="cancel jobs")
    cancel.add_argument("jobs", nargs="*", metavar="JOB", help="job ids or ranges such as 300-440")
    cancel.add_argument("--group", metavar="NAME", help="jobs in this group")
    cancel_which = cancel.add_mutually_exclusive_group()
    cancel_which.add_argument(
        "--waiting", action="store_true", help="only jobs that have not started"
    )
    cancel_which.add_argument(
        "--state",
        action="append",
        choices=[JobState.WAITING.value, JobState.STARTING.value, JobState.RUNNING.value],
        help="only jobs in this state (repeatable)",
    )
    cancel.add_argument(
        "-y", "--yes", action="store_true", help="do not ask before killing running jobs"
    )

    retry = sub.add_parser("retry", help="resubmit failed or cancelled jobs")
    retry.add_argument("jobs", nargs="*", metavar="JOB", help="job ids or ranges such as 216-218")
    retry.add_argument("--group", metavar="NAME", help="jobs in this group")
    retry_which = retry.add_mutually_exclusive_group()
    retry_which.add_argument("--failed", action="store_true", help="only FAILED jobs")
    retry_which.add_argument("--cancelled", action="store_true", help="only CANCELLED jobs")

    update = sub.add_parser("update", help="upgrade gq to the latest release")
    update.add_argument(
        "--check", action="store_true", help="only report whether an update is available"
    )
    when = update.add_mutually_exclusive_group()
    when.add_argument(
        "--wait",
        action="store_true",
        help="pause the queue and let running jobs finish before updating",
    )
    when.add_argument(
        "--force",
        action="store_true",
        help="update while jobs run; they are recorded as FAILED when they finish",
    )

    daemon = sub.add_parser("daemon", help="manage the scheduler daemon")
    daemon.add_argument(
        "action",
        choices=("start", "stop", "status", "pause", "resume"),
        help="pause stops queued jobs from starting; running jobs are unaffected",
    )
    daemon.add_argument(
        "--group", metavar="NAME", help="with pause or resume: hold or release only this group"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        code = _main(argv)
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        # The reader went away (`gq ps | head`). Point stdout at /dev/null so the
        # interpreter's final flush cannot print a second error, and exit the way a
        # process killed by SIGPIPE would.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 141


def _main(argv: list[str] | None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and (
            arguments[0] in {"-g", "--gpus"}
            or (arguments[0].startswith("-g") and arguments[0] != "--help")
            or arguments[0].startswith("--gpus=")
        ):
            args = _run_parser("gq").parse_args(arguments)
            _submit(
                args.command,
                args.gpus,
                args.name,
                env_all=args.env_all,
                env_keep=args.env_keep,
                group=args.group,
                key=args.key,
            )
            return 0

        args = _main_parser().parse_args(arguments)
        if args.subcommand == "run":
            _submit(
                args.command,
                args.gpus,
                args.name,
                env_all=args.env_all,
                env_keep=args.env_keep,
                group=args.group,
                key=args.key,
            )
        elif args.subcommand == "submit":
            script = args.script.expanduser().resolve()
            if not script.is_file():
                raise CLIError(f"script does not exist or is not a regular file: {script}")
            directives = _script_directives(script)
            _submit(
                ["/bin/bash", str(script)],
                args.gpus if args.gpus is not None else directives.gpus or 1,
                args.name if args.name is not None else directives.name,
                env_all=args.env_all,
                env_keep=args.env_keep,
                group=args.group if args.group is not None else directives.group,
                key=args.key if args.key is not None else directives.key,
            )
        elif args.subcommand == "ps":
            _print_jobs(
                args.limit, args.json, tokens=args.jobs, group=args.group, states=args.state
            )
        elif args.subcommand == "groups":
            _print_groups(args.json)
        elif args.subcommand == "gpu":
            _print_gpus(args.json)
        elif args.subcommand == "show":
            _show(args.job_id, args.json)
        elif args.subcommand == "rm":
            _remove_jobs(args.job_id)
        elif args.subcommand == "clean":
            _clean_jobs(args.older_than, args.state, args.yes)
        elif args.subcommand == "logs":
            _logs(args.job_id, args.follow)
        elif args.subcommand == "cancel":
            states = [JobState.WAITING.value] if args.waiting else args.state
            _cancel(args.jobs, args.group, states, args.yes)
        elif args.subcommand == "retry":
            states = (
                [JobState.FAILED.value]
                if args.failed
                else [JobState.CANCELLED.value]
                if args.cancelled
                else None
            )
            _retry(args.jobs, args.group, states)
        elif args.subcommand == "update":
            return _update(args.check, args.wait, args.force)
        elif args.subcommand == "daemon":
            paths = Paths.from_environment()
            if args.action == "start":
                _start_daemon(paths)
            elif args.action == "stop":
                _stop_daemon(paths)
            elif args.group is not None and args.action in ("pause", "resume"):
                response = _daemon_request({"type": f"{args.action}_queue", "group": args.group})
                if args.action == "pause":
                    print(
                        f"Group {args.group} is held; its queued jobs will not start."
                        if response["changed"]
                        else f"Group {args.group} was already held."
                    )
                else:
                    print(
                        f"Released the hold on group {args.group}."
                        if response["changed"]
                        else f"Group {args.group} was not held."
                    )
            elif args.group is not None:
                raise CLIError("--group only applies to 'gq daemon pause' and 'gq daemon resume'")
            elif args.action == "pause":
                response = _daemon_request({"type": "pause_queue"})
                active = response["active_job_ids"]
                print(
                    "Queue paused; queued jobs will not start until 'gq daemon resume'. "
                    + (f"Still running: {', '.join(map(str, active))}." if active else "")
                )
            elif args.action == "resume":
                _daemon_request({"type": "resume_queue"})
                print("Queue resumed.")
            else:
                running = _ping(paths)
                if running:
                    print(
                        f"gq daemon is running (pid {running['pid']}, version {running['version']})"
                    )
                    if running.get("queue_paused"):
                        print("the queue is paused; resume it with 'gq daemon resume'")
                    spoken = running.get("protocol")
                    if spoken != PROTOCOL_VERSION:
                        print(
                            f"warning: daemon speaks protocol "
                            f"{spoken if spoken is not None else 'unknown (pre-0.1.1)'} "
                            f"but this client speaks {PROTOCOL_VERSION}; run "
                            "'gq daemon stop' and let the next command restart it",
                            file=sys.stderr,
                        )
                else:
                    print("gq daemon is not running")
                    return 1
        return 0
    except (CLIError, ProtocolError) as exc:
        print(f"gq: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
