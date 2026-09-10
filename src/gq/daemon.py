from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any

from . import __version__
from .database import Database
from .models import JobState
from .nvml import NVMLProvider
from .paths import Paths
from .protocol import PROTOCOL_VERSION
from .scheduler import Scheduler, SchedulerError


class AlreadyRunning(RuntimeError):
    pass


def in_container() -> bool:
    """Best-effort container detection, used only to soften the root check.

    Getting this wrong in either direction is not a safety problem: the root guard
    protects a shared host from a single-user scheduler running with more privilege
    than it needs, and `GQ_ALLOW_ROOT=1` remains the explicit override.
    """
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    # Lowercase by convention: systemd-nspawn and podman set `container`, not `CONTAINER`.
    if os.environ.get("container"):  # noqa: SIM112
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
    except OSError:
        return False
    # cgroup v2 hosts often report a bare "0::/", so this only adds v1-style hits.
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd", "lxc"))


class Daemon:
    def __init__(self, paths: Paths, reconcile_interval: float = 2.0) -> None:
        self.paths = paths
        self.reconcile_interval = reconcile_interval
        self.stop_event = asyncio.Event()
        self.lock_handle: Any = None
        self.database: Database | None = None
        self.scheduler: Scheduler | None = None
        self.server: asyncio.AbstractServer | None = None
        self.log = logging.getLogger("gq.daemon")

    def acquire_instance_lock(self) -> None:
        self.paths.ensure()
        self.lock_handle = self.paths.lock_file.open("a+")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_handle.close()
            raise AlreadyRunning("another gq daemon holds the instance lock") from exc
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(str(os.getpid()))
        self.lock_handle.flush()

    async def run(self) -> None:
        if os.geteuid() == 0 and os.environ.get("GQ_ALLOW_ROOT") != "1":
            if in_container():
                self.log.info(
                    "running as root inside a container; this is expected for "
                    "Docker/Jupyter images where uid 0 is the only account"
                )
            else:
                raise RuntimeError(
                    "refusing to run the user scheduler as root on what looks like a "
                    "regular host, where jobs would gain unnecessary privilege. "
                    "Run gq as your own user, or set GQ_ALLOW_ROOT=1 to override."
                )
        self.acquire_instance_lock()
        # The lock proves no live daemon owns this path, so removing a stale socket is safe.
        try:
            self.paths.socket.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot remove stale socket {self.paths.socket}: {exc}") from exc
        self.paths.pid_file.write_text(f"{os.getpid()}\n")
        os.chmod(self.paths.pid_file, 0o600)

        self.database = Database(self.paths.database)
        provider = NVMLProvider()
        self.scheduler = Scheduler(
            self.database,
            provider,
            self.paths.logs_dir,
            cancel_grace_seconds=float(os.environ.get("GQ_CANCEL_GRACE_SECONDS", "10")),
            logger=self.log,
        )
        await self.scheduler.initialize()
        self.server = await asyncio.start_unix_server(
            self._handle_client, path=self.paths.socket, limit=4 * 1024 * 1024
        )
        os.chmod(self.paths.socket, 0o600)
        self.log.info("gq daemon %s started (pid=%s)", __version__, os.getpid())
        reconcile_task = asyncio.create_task(self._reconciliation_loop())
        try:
            await self.stop_event.wait()
        finally:
            self.log.info("gq daemon stopping; managed jobs will continue running")
            self.server.close()
            await self.server.wait_closed()
            reconcile_task.cancel()
            await asyncio.gather(reconcile_task, return_exceptions=True)
            await self.scheduler.close()
            self.database.close()
            self.paths.socket.unlink(missing_ok=True)
            self.paths.pid_file.unlink(missing_ok=True)
            if self.lock_handle:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
                self.lock_handle.close()

    async def _reconciliation_loop(self) -> None:
        assert self.scheduler is not None
        while True:
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.reconcile_interval)
                return
            except TimeoutError:
                try:
                    await self.scheduler.tick()
                except Exception:
                    self.log.exception("reconciliation iteration failed")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await reader.readline()
            if not raw:
                return
            if len(raw) > 4 * 1024 * 1024:
                response = {"ok": False, "error": "request is too large"}
            else:
                try:
                    request = json.loads(raw)
                    if not isinstance(request, dict):
                        raise ValueError("request must be a JSON object")
                    response = await self._dispatch(request)
                except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                    response = {"ok": False, "error": f"invalid request: {exc}"}
                except SchedulerError as exc:
                    response = {"ok": False, "error": str(exc)}
                except Exception:
                    self.log.exception("request failed")
                    response = {"ok": False, "error": "internal daemon error; see daemon log"}
            writer.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError):
                # A client that stops reading and hangs up is ordinary, not a daemon
                # error: `gq ps | head`, or Ctrl-C out of `gq logs -f`. Letting this
                # escape logged a full traceback per occurrence.
                return
        finally:
            writer.close()
            with contextlib.suppress(ConnectionResetError, BrokenPipeError):
                await writer.wait_closed()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.scheduler is not None
        kind = request.get("type")
        client_protocol = request.get("protocol")
        if client_protocol is not None and client_protocol != PROTOCOL_VERSION:
            raise SchedulerError(
                f"gq client speaks protocol {client_protocol} but this daemon speaks "
                f"{PROTOCOL_VERSION}; run 'gq daemon stop' then retry to restart it "
                "at the installed version"
            )
        if kind == "ping":
            return {
                "ok": True,
                "pid": os.getpid(),
                "version": __version__,
                "protocol": PROTOCOL_VERSION,
            }
        if kind == "submit":
            argv = request["argv"]
            env = request["env"]
            if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
                raise ValueError("argv must be a list of strings")
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()
            ):
                raise ValueError("env must be an object containing strings")
            job = await self.scheduler.submit(
                argv,
                Path(request["cwd"]),
                env,
                int(request["requested_gpus"]),
                request.get("name"),
            )
            return {"ok": True, "job_id": job.id, "state": job.state.value}
        if kind == "list_jobs":
            limit = request.get("limit")
            jobs = await self.scheduler.list_jobs(int(limit) if limit is not None else None)
            return {"ok": True, "jobs": [job.to_dict() for job in jobs]}
        if kind == "show_job":
            job = await self.scheduler.get_job(int(request["job_id"]))
            if job is None:
                raise SchedulerError(f"job {request['job_id']} does not exist")
            return {"ok": True, "job": job.to_dict()}
        if kind == "gpu_status":
            statuses = await self.scheduler.gpu_status()
            return {"ok": True, "gpus": [status.to_dict() for status in statuses]}
        if kind == "cancel_job":
            message = await self.scheduler.cancel(int(request["job_id"]))
            return {"ok": True, "message": message}
        if kind == "delete_jobs":
            raw_ids = request["job_ids"]
            if not isinstance(raw_ids, list) or not raw_ids:
                raise ValueError("job_ids must be a non-empty list")
            removed = await self.scheduler.delete_jobs([int(v) for v in raw_ids])
            return {"ok": True, "removed": [job.id for job in removed]}
        if kind == "clean_jobs":
            raw_states = request.get("states")
            states: list[JobState] | None = None
            if raw_states is not None:
                if not isinstance(raw_states, list):
                    raise ValueError("states must be a list")
                try:
                    states = [JobState(value) for value in raw_states]
                except ValueError as exc:
                    raise ValueError(f"unknown job state: {exc}") from exc
            removed = await self.scheduler.clean_jobs(str(request["cutoff"]), states)
            return {"ok": True, "removed": [job.id for job in removed]}
        if kind == "shutdown":
            self.stop_event.set()
            return {"ok": True, "message": "daemon stopping; running jobs are unchanged"}
        raise ValueError(f"unknown request type: {kind!r}")


def configure_logging(path: Path, foreground: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.FileHandler(path)]
    if foreground:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gq scheduler daemon")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args(argv)
    paths = Paths.from_environment()
    paths.ensure()
    configure_logging(paths.daemon_log, args.foreground)
    daemon = Daemon(
        paths,
        reconcile_interval=float(os.environ.get("GQ_RECONCILE_INTERVAL", "2")),
    )

    async def run() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, daemon.stop_event.set)
        await daemon.run()

    try:
        asyncio.run(run())
    except AlreadyRunning as exc:
        logging.error("%s", exc)
        return 1
    except Exception:
        logging.exception("daemon terminated with an error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
