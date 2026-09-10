from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .models import JobState
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
) -> None:
    if not argv:
        raise CLIError("a command is required")
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
        }
    )
    print(f"Submitted job {response['job_id']}")


DIRECTIVE_RE = re.compile(r"^\s*#gq\s+(--(?:gpus|name))=(.*)\s*$")


def _script_directives(path: Path) -> tuple[int | None, str | None]:
    try:
        text = path.read_text()
    except (OSError, UnicodeError) as exc:
        raise CLIError(f"cannot read script {path}: {exc}") from exc
    gpus: int | None = None
    name: str | None = None
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
                raise CLIError(f"empty job name on line {line_number}")
            name = value
    return gpus, name


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


def _print_jobs(limit: int | None, as_json: bool = False) -> None:
    response = _daemon_request({"type": "list_jobs", "limit": limit})
    if as_json:
        print(json.dumps(response["jobs"], indent=2))
        return
    rows = []
    for job in response["jobs"]:
        gpus = ",".join(map(str, job["gpu_indices"])) or "-"
        rows.append(
            (
                job["id"],
                job["state"],
                gpus,
                job["pid"] or "-",
                job["name"] or "-",
                _command_text(job["argv"]),
            )
        )
    print(_table(["JOB", "STATE", "GPUs", "PID", "NAME", "COMMAND"], rows))


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
    job = response["job"]
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


def _run_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="submit a GPU job")
    parser.add_argument("-g", "--gpus", type=int, required=True, help="number of GPUs")
    parser.add_argument("--name", help="job name")
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
    _add_env_options(run)
    run.add_argument("command", nargs=argparse.REMAINDER)

    submit = sub.add_parser("submit", help="submit a shell script")
    submit.add_argument("script", type=Path)
    submit.add_argument("-g", "--gpus", type=int)
    submit.add_argument("--name")
    _add_env_options(submit)

    ps = sub.add_parser("ps", help="list jobs")
    ps.add_argument("--limit", type=int)
    ps.add_argument("--json", action="store_true", help="emit JSON instead of a table")
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

    cancel = sub.add_parser("cancel", help="cancel a job")
    cancel.add_argument("job_id", type=int)

    daemon = sub.add_parser("daemon", help="manage the scheduler daemon")
    daemon.add_argument("action", choices=("start", "stop", "status"))
    return parser


def main(argv: list[str] | None = None) -> int:
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
            )
        elif args.subcommand == "submit":
            script = args.script.expanduser().resolve()
            if not script.is_file():
                raise CLIError(f"script does not exist or is not a regular file: {script}")
            directive_gpus, directive_name = _script_directives(script)
            _submit(
                ["/bin/bash", str(script)],
                args.gpus if args.gpus is not None else directive_gpus or 1,
                args.name if args.name is not None else directive_name,
                env_all=args.env_all,
                env_keep=args.env_keep,
            )
        elif args.subcommand == "ps":
            _print_jobs(args.limit, args.json)
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
            response = _daemon_request({"type": "cancel_job", "job_id": args.job_id})
            print(f"Job {args.job_id}: {response['message']}")
        elif args.subcommand == "daemon":
            paths = Paths.from_environment()
            if args.action == "start":
                _start_daemon(paths)
            elif args.action == "stop":
                _stop_daemon(paths)
            else:
                running = _ping(paths)
                if running:
                    print(
                        f"gq daemon is running (pid {running['pid']}, version {running['version']})"
                    )
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
