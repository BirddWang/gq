from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from datetime import datetime
from pathlib import Path

from .allocator import classify_gpus, select_gpus
from .database import Database, KeyInUse
from .models import MANUAL_HOLD_REASON, GPUObservation, GPUState, GPUStatus, Job, JobState
from .nvml import GPUProvider
from .processes import (
    boot_id,
    managed_process_group_alive,
    pid_in_process_group,
    process_group_members,
    process_start_time,
    signal_managed_group,
)


class SchedulerError(RuntimeError):
    pass


class DuplicateKey(SchedulerError):
    """A submission was skipped because its key is held by a live or successful job."""

    def __init__(self, existing: Job) -> None:
        state = existing.state.value
        super().__init__(f"key {existing.key!r} is already {state} as job {existing.id}")
        self.existing = existing


MAX_LABEL_LENGTH = 200


def _validate_label(value: str | None, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SchedulerError(f"{what} must be a non-empty string")
    if len(value) > MAX_LABEL_LENGTH:
        raise SchedulerError(f"{what} must be at most {MAX_LABEL_LENGTH} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise SchedulerError(f"{what} must not contain control characters")
    return value


def _runtime(job: Job) -> float:
    """Seconds from reservation to finish; infinite when either end is unknown."""
    if job.start_time is None or job.end_time is None:
        return float("inf")
    return (
        datetime.fromisoformat(job.end_time) - datetime.fromisoformat(job.start_time)
    ).total_seconds()


class Scheduler:
    """The daemon's sole mutation context for jobs and GPU allocations."""

    def __init__(
        self,
        database: Database,
        gpu_provider: GPUProvider,
        logs_dir: Path,
        *,
        cancel_grace_seconds: float = 10.0,
        fail_fast_count: int = 3,
        fail_fast_seconds: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db = database
        # A group is held once this many of its jobs in a row fail within
        # fail_fast_seconds of starting. Zero disables holding.
        self.fail_fast_count = fail_fast_count
        self.fail_fast_seconds = fail_fast_seconds
        self.gpu_provider = gpu_provider
        self.logs_dir = logs_dir
        self.cancel_grace_seconds = cancel_grace_seconds
        self.log = logger or logging.getLogger("gq.scheduler")
        self.lock = asyncio.Lock()
        self._observations: list[GPUObservation] = []
        self._snapshot_error: str | None = None
        self._processes: dict[int, asyncio.subprocess.Process] = {}
        self._monitor_tasks: dict[int, asyncio.Task[None]] = {}
        self._cancellation_tasks: dict[int, asyncio.Task[None]] = {}
        self._stopping = False
        # In memory only, deliberately: a daemon restart always resumes the queue,
        # which is what an update that paused it wants.
        self._paused = False
        # Mirrors the group_holds table. This daemon is its only writer, and the
        # scheduling loop consults it before every launch.
        self._held: dict[str, str] = {}

    @property
    def paused(self) -> bool:
        return self._paused

    async def set_paused(self, paused: bool) -> None:
        """Stop or restart launching queued jobs. Running jobs are never affected."""
        async with self.lock:
            if paused == self._paused:
                return
            self._paused = paused
            if paused:
                self.log.info("queue paused: running jobs continue, queued jobs will wait")
            else:
                self.log.info("queue resumed")
                await self._try_schedule_locked()

    async def initialize(self) -> None:
        async with self.lock:
            self._held = self.db.held_groups()
            self._refresh_observations_locked()
            await self._recover_locked()
            await self._try_schedule_locked()

    async def submit(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        requested_gpus: int,
        name: str | None = None,
        *,
        group: str | None = None,
        key: str | None = None,
    ) -> Job:
        group = _validate_label(group, "group")
        key = _validate_label(key, "key")
        if not argv or not argv[0]:
            raise SchedulerError("command must not be empty")
        if requested_gpus < 1:
            raise SchedulerError("requested GPU count must be at least 1")
        if not cwd.is_dir():
            raise SchedulerError(f"working directory does not exist: {cwd}")
        if any(not isinstance(item, str) or "\x00" in item for item in argv):
            raise SchedulerError("command contains an invalid argument")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
            raise SchedulerError("environment keys and values must be strings")

        async with self.lock:
            self._refresh_observations_locked()
            if self._snapshot_error and not self._observations:
                raise SchedulerError(f"cannot inspect GPUs: {self._snapshot_error}")
            gpu_count = len(self._observations)
            if requested_gpus > gpu_count:
                raise SchedulerError(
                    f"requested {requested_gpus} GPUs, but this machine has "
                    f"{gpu_count} visible GPUs"
                )
            job = self._create_locked(
                argv,
                cwd.resolve(),
                env,
                requested_gpus,
                name or Path(argv[0]).name,
                group=group,
                key=key,
            )
            self.log.info("submitted job %s requesting %s GPU(s): %r", job.id, requested_gpus, argv)
            await self._try_schedule_locked()
            result = self.db.get_job(job.id)
            assert result is not None
            return result

    async def tick(self) -> None:
        async with self.lock:
            self._refresh_observations_locked()
            await self._reap_recovered_jobs_locked()
            await self._try_schedule_locked()

    async def active_jobs(self) -> list[Job]:
        async with self.lock:
            return self.db.active_jobs()

    async def list_jobs(
        self,
        limit: int | None = None,
        *,
        group: str | None = None,
        states: list[JobState] | None = None,
        id_ranges: list[tuple[int, int]] | None = None,
    ) -> list[Job]:
        async with self.lock:
            return self.db.list_jobs(limit, group=group, states=states, id_ranges=id_ranges)

    async def get_job(self, job_id: int) -> Job | None:
        async with self.lock:
            return self.db.get_job(job_id)

    async def gpu_status(self, *, refresh: bool = True) -> list[GPUStatus]:
        async with self.lock:
            if refresh:
                self._refresh_observations_locked()
            return self._gpu_status_locked()

    async def cancel(self, job_id: int) -> str:
        async with self.lock:
            job = self.db.get_job(job_id)
            if job is None:
                raise SchedulerError(f"job {job_id} does not exist")
            if job.state.terminal:
                raise SchedulerError(f"job {job_id} is already {job.state.value}")
            outcome = self._cancel_locked(job)
            await self._try_schedule_locked()
            return outcome

    async def cancel_many(self, job_ids: list[int]) -> dict[str, list[int]]:
        """Cancel a batch under one lock hold.

        Separate per-job requests would leave gaps in which a running job exits and its
        GPU is handed to a waiting job that a later request in the batch then kills.
        """
        result: dict[str, list[int]] = {"cancelled": [], "cancelling": [], "skipped": []}
        async with self.lock:
            for job_id in job_ids:
                job = self.db.get_job(job_id)
                if job is None or job.state.terminal:
                    result["skipped"].append(job_id)
                    continue
                outcome = self._cancel_locked(job)
                result["cancelled" if outcome == "cancelled" else "cancelling"].append(job_id)
            await self._try_schedule_locked()
        return result

    def _cancel_locked(self, job: Job) -> str:
        if job.state is JobState.WAITING:
            self.db.finish(job.id, JobState.CANCELLED, reason="cancelled before launch")
            self.log.info("cancelled waiting job %s", job.id)
            return "cancelled"
        if job.state is JobState.CANCELLING:
            return "cancellation already in progress"
        if not self.db.set_cancelling(job.id):
            raise SchedulerError(f"job {job.id} cannot be cancelled from {job.state.value}")
        sent = signal_managed_group(job.pid, job.pgid, job.process_start_time, signal.SIGTERM)
        self.log.info("cancelling job %s with SIGTERM (sent=%s)", job.id, sent)
        task = asyncio.create_task(self._finish_cancellation(job.id))
        self._cancellation_tasks[job.id] = task
        return "cancellation requested"

    async def retry(self, job_ids: list[int]) -> dict[str, object]:
        """Resubmit failed or cancelled jobs with their original command and context.

        Only the newest attempt in a retry lineage is retried, so running the same retry
        twice cannot queue duplicates.
        """
        created: list[dict[str, int]] = []
        skipped: list[dict[str, object]] = []
        released: list[str] = []
        async with self.lock:
            self._refresh_observations_locked()
            gpu_count = len(self._observations)
            for job_id in job_ids:
                job = self.db.get_job(job_id)
                reason = None
                if job is None:
                    reason = "does not exist"
                elif job.state not in (JobState.FAILED, JobState.CANCELLED):
                    reason = f"is {job.state.value}; only FAILED and CANCELLED jobs are retried"
                elif (newer := self.db.newer_attempt(job.retry_of or job.id, job.id)) is not None:
                    reason = f"was already retried as job {newer}"
                elif not job.cwd.is_dir():
                    reason = f"its working directory no longer exists: {job.cwd}"
                elif job.requested_gpus > gpu_count:
                    reason = f"needs {job.requested_gpus} GPUs but {gpu_count} are visible"
                if job is None or reason is not None:
                    skipped.append({"id": job_id, "reason": reason})
                    continue
                try:
                    attempt = self._create_locked(
                        job.argv,
                        job.cwd,
                        job.env,
                        job.requested_gpus,
                        job.name,
                        group=job.group,
                        key=job.key,
                        retry_of=job.retry_of or job.id,
                    )
                except DuplicateKey as exc:
                    skipped.append({"id": job_id, "reason": str(exc)})
                    continue
                created.append({"from": job.id, "job_id": attempt.id})
                self.log.info("retrying job %s as job %s", job.id, attempt.id)
                # Retrying says the cause was addressed, so a hold on the group would
                # only stop the retry from running.
                if job.group in self._held and self._set_hold_locked(job.group, False):
                    released.append(job.group)
            await self._try_schedule_locked()
        return {"created": created, "skipped": skipped, "released_groups": released}

    async def set_group_held(self, group: str, held: bool, reason: str | None = None) -> bool:
        group = _validate_label(group, "group") or ""
        async with self.lock:
            changed = self._set_hold_locked(group, held, reason or MANUAL_HOLD_REASON)
            if changed and not held:
                await self._try_schedule_locked()
            return changed

    async def group_summaries(self) -> list[dict[str, object]]:
        async with self.lock:
            counts = self.db.group_counts()
            retried = self.db.group_retried_counts()
            names = sorted(set(counts) | set(retried) | set(self._held))
            return [
                {
                    "group": name,
                    "counts": counts.get(name, {}),
                    "retried": retried.get(name, 0),
                    "held": name in self._held,
                    "hold_reason": self._held.get(name),
                }
                for name in names
            ]

    @property
    def held_groups(self) -> dict[str, str]:
        return dict(self._held)

    def _set_hold_locked(self, group: str, held: bool, reason: str | None = None) -> bool:
        changed = self.db.set_group_hold(group, held, reason if held else None)
        if held:
            self._held[group] = reason or ""
        else:
            self._held.pop(group, None)
        if changed:
            if held:
                self.log.warning("holding group %s: %s", group, reason)
            else:
                self.log.info("released hold on group %s", group)
        return changed

    def _create_locked(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        requested_gpus: int,
        name: str | None,
        *,
        group: str | None,
        key: str | None,
        retry_of: int | None = None,
    ) -> Job:
        try:
            job = self.db.create_job(
                argv,
                cwd,
                env,
                requested_gpus,
                name,
                self.logs_dir / "pending.log",
                group=group,
                key=key,
                retry_of=retry_of,
            )
        except KeyInUse as exc:
            existing = self.db.get_job(exc.job_id)
            assert existing is not None
            raise DuplicateKey(existing) from exc
        self.db.set_log_path(job.id, self.logs_dir / f"{job.id}.log")
        refreshed = self.db.get_job(job.id)
        assert refreshed is not None
        return refreshed

    def _note_failure_locked(self, job_id: int) -> None:
        """Hold a job's group if its recent jobs keep failing right after they start.

        Called only for failures of jobs this daemon launched and watched, never for jobs
        failed by restart recovery, whose timing says nothing about the command.
        """
        if self.fail_fast_count <= 0:
            return
        job = self.db.get_job(job_id)
        if job is None or job.group is None or job.group in self._held:
            return
        since = self.db.group_released_at(job.group)
        recent = self.db.recent_group_outcomes(job.group, since, self.fail_fast_count)
        if len(recent) < self.fail_fast_count:
            return
        if all(
            item.state is JobState.FAILED and _runtime(item) < self.fail_fast_seconds
            for item in recent
        ):
            ids = ", ".join(str(item.id) for item in sorted(recent, key=lambda j: j.id))
            self._set_hold_locked(
                job.group,
                True,
                f"{len(recent)} jobs in a row failed within {self.fail_fast_seconds:g}s "
                f"of starting ({ids})",
            )

    async def delete_jobs(self, job_ids: list[int]) -> list[Job]:
        async with self.lock:
            try:
                removed = self.db.delete_jobs(job_ids)
            except (KeyError, ValueError) as exc:
                raise SchedulerError(str(exc)) from exc
        self._remove_logs(removed)
        return removed

    async def clean_jobs(self, cutoff: str, states: list[JobState] | None = None) -> list[Job]:
        async with self.lock:
            stale = self.db.terminal_jobs_before(cutoff, states)
            removed = self.db.delete_jobs([job.id for job in stale])
        self._remove_logs(removed)
        return removed

    def _remove_logs(self, jobs: list[Job]) -> None:
        logs_dir = self.logs_dir.resolve()
        for job in jobs:
            if job.log_path is None:
                continue
            try:
                # Only ever unlink inside the managed log directory. A hand-edited or
                # corrupted log_path must not turn cleanup into arbitrary deletion.
                job.log_path.resolve().relative_to(logs_dir)
            except (ValueError, OSError):
                self.log.warning(
                    "refusing to remove log outside %s for job %s: %s",
                    logs_dir,
                    job.id,
                    job.log_path,
                )
                continue
            try:
                job.log_path.unlink(missing_ok=True)
            except OSError as exc:
                self.log.warning("could not remove log for job %s: %s", job.id, exc)

    async def close(self) -> None:
        # Wait for an in-flight submission/launch to finish before closing NVML.
        async with self.lock:
            self._stopping = True
            tasks = [*self._monitor_tasks.values(), *self._cancellation_tasks.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.gpu_provider.close()

    def _refresh_observations_locked(self) -> None:
        try:
            observations = self.gpu_provider.snapshot()
        except Exception as exc:
            # Keep device identity from the last successful scan, but mark every
            # device UNKNOWN until NVML recovers.
            self._snapshot_error = str(exc)
            self.log.warning("GPU observation failed: %s", exc)
            self._observations = [
                GPUObservation(
                    uuid=item.uuid,
                    index=item.index,
                    name=item.name,
                    total_memory=item.total_memory,
                    used_memory=item.used_memory,
                    error=f"NVML unavailable: {exc}",
                )
                for item in self._observations
            ]
        else:
            self._observations = observations
            self._snapshot_error = None

    def _gpu_status_locked(self) -> list[GPUStatus]:
        assignments = self.db.assignments()
        assignment_jobs = set(assignments.values())
        active = {
            job_id: job
            for job_id in assignment_jobs
            if (job := self.db.get_job(job_id)) is not None
        }
        return classify_gpus(
            self._observations,
            assignments,
            active,
            lambda pid, job: pid_in_process_group(pid, job.pgid),
        )

    async def _try_schedule_locked(self) -> None:
        if self._stopping or self._paused:
            return
        statuses = self._gpu_status_locked()
        for waiting in self.db.waiting_jobs():
            # Re-checked for every job: a launch failure earlier in this same pass can
            # put the group on hold.
            if waiting.group is not None and waiting.group in self._held:
                continue
            selected = select_gpus(statuses, waiting.requested_gpus)
            if not selected:
                continue  # simple backfilling: try younger jobs
            observations_by_uuid = {gpu.uuid: gpu for gpu in self._observations}
            selected_observations = [observations_by_uuid[gpu.uuid] for gpu in selected]
            self.db.reserve(waiting.id, selected_observations)
            # Remove all selected devices before any subsequent allocation. The
            # DB reservation is already durable if launch fails or the daemon dies.
            selected_uuids = {gpu.uuid for gpu in selected}
            statuses = [
                GPUStatus(
                    **{
                        **status.__dict__,
                        "state": GPUState.RESERVED,
                        "owner_job_id": waiting.id,
                    }
                )
                if status.uuid in selected_uuids
                else status
                for status in statuses
            ]
            await self._launch_locked(waiting.id, selected)
            # A launch failure releases GPUs; rebuild before considering the next job.
            statuses = self._gpu_status_locked()

    async def _launch_locked(self, job_id: int, selected: list[GPUStatus]) -> None:
        job = self.db.get_job(job_id)
        assert job is not None and job.state is JobState.STARTING
        assert job.log_path is not None
        env = dict(job.env)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu.index) for gpu in selected)
        process: asyncio.subprocess.Process | None = None
        try:
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = job.log_path.open("ab", buffering=0)
            try:
                process = await asyncio.create_subprocess_exec(
                    *job.argv,
                    cwd=job.cwd,
                    env=env,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                log_handle.close()
            # start_new_session makes the child its process-group leader. Using
            # the known PID avoids a race with ultrashort commands exiting before
            # a separate getpgid() call.
            pgid = process.pid
            started = process_start_time(process.pid)
            if started is None:
                try:
                    exit_code = await asyncio.wait_for(process.wait(), timeout=0.1)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pgid, signal.SIGKILL)
                    exit_code = await process.wait()
                    self.db.finish(
                        job.id,
                        JobState.FAILED,
                        exit_code=exit_code,
                        reason="could not record Linux process identity after launch",
                    )
                    self._note_failure_locked(job.id)
                else:
                    state = JobState.DONE if exit_code == 0 else JobState.FAILED
                    reason = None if exit_code == 0 else f"command exited with status {exit_code}"
                    self.db.finish(job.id, state, exit_code=exit_code, reason=reason)
                    if state is JobState.FAILED:
                        self._note_failure_locked(job.id)
                return
            self.db.mark_running(job.id, process.pid, pgid, started, boot_id())
            self._processes[job.id] = process
            self._monitor_tasks[job.id] = asyncio.create_task(
                self._monitor_process(job.id, process)
            )
            self.log.info(
                "started job %s pid=%s GPUs=%s",
                job.id,
                process.pid,
                ",".join(str(gpu.index) for gpu in selected),
            )
        except Exception as exc:
            if process is not None and process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            self.db.finish(
                job.id,
                JobState.FAILED,
                reason=f"process launch failed: {exc}",
            )
            self.log.exception("failed to launch job %s", job.id)
            self._note_failure_locked(job.id)

    async def _monitor_process(self, job_id: int, process: asyncio.subprocess.Process) -> None:
        try:
            exit_code = await process.wait()
            job = self.db.get_job(job_id)
            if job and job.pgid is not None:
                # A launcher can exit before workers. Keep the reservation until
                # the full managed process group is gone.
                while process_group_members(job.pgid):
                    await asyncio.sleep(0.25)
            async with self.lock:
                job = self.db.get_job(job_id)
                if job is None or job.state.terminal:
                    return
                if job.state is JobState.CANCELLING:
                    final_state = JobState.CANCELLED
                    reason = "cancelled by user"
                elif exit_code == 0:
                    final_state = JobState.DONE
                    reason = None
                else:
                    final_state = JobState.FAILED
                    reason = f"command exited with status {exit_code}"
                self.db.finish(job_id, final_state, exit_code=exit_code, reason=reason)
                self.log.info(
                    "job %s finished as %s (exit=%s)", job_id, final_state.value, exit_code
                )
                if final_state is JobState.FAILED:
                    self._note_failure_locked(job_id)
                self._refresh_observations_locked()
                await self._try_schedule_locked()
        except asyncio.CancelledError:
            raise
        finally:
            self._processes.pop(job_id, None)
            self._monitor_tasks.pop(job_id, None)

    async def _finish_cancellation(self, job_id: int) -> None:
        try:
            deadline = asyncio.get_running_loop().time() + self.cancel_grace_seconds
            while asyncio.get_running_loop().time() < deadline:
                job = self.db.get_job(job_id)
                if job is None or not managed_process_group_alive(
                    job.pid, job.pgid, job.process_start_time
                ):
                    break
                await asyncio.sleep(0.1)
            else:
                job = self.db.get_job(job_id)
                if job:
                    sent = signal_managed_group(
                        job.pid, job.pgid, job.process_start_time, signal.SIGKILL
                    )
                    self.log.warning("escalated cancellation of job %s (sent=%s)", job_id, sent)

            process = self._processes.get(job_id)
            if process is not None:
                return  # the normal monitor owns the terminal transition

            # Recovered jobs have no asyncio Process to monitor.
            while True:
                job = self.db.get_job(job_id)
                if job is None or not managed_process_group_alive(
                    job.pid, job.pgid, job.process_start_time
                ):
                    break
                await asyncio.sleep(0.1)
            async with self.lock:
                job = self.db.get_job(job_id)
                if job and job.state is JobState.CANCELLING:
                    self.db.finish(job_id, JobState.CANCELLED, reason="cancelled by user")
                    self._refresh_observations_locked()
                    await self._try_schedule_locked()
        finally:
            self._cancellation_tasks.pop(job_id, None)

    async def _recover_locked(self) -> None:
        current_boot = boot_id()
        known_uuids = {gpu.uuid for gpu in self._observations}
        for job in self.db.active_jobs():
            reason: str | None = None
            terminal = JobState.FAILED
            if job.boot_id and current_boot and job.boot_id != current_boot:
                reason = "interrupted by machine reboot"
            elif not managed_process_group_alive(job.pid, job.pgid, job.process_start_time):
                reason = "managed process was not alive during daemon recovery"
                if job.state is JobState.CANCELLING:
                    terminal = JobState.CANCELLED
            elif not job.gpu_uuids or any(uuid not in known_uuids for uuid in job.gpu_uuids):
                reason = "saved GPU allocation is missing after daemon restart"
                terminal = JobState.ORPHANED
            if reason:
                self.db.finish(
                    job.id,
                    terminal,
                    reason=reason,
                )
                self.log.warning("recovered job %s as %s: %s", job.id, terminal.value, reason)
            elif job.state is JobState.CANCELLING:
                self._cancellation_tasks[job.id] = asyncio.create_task(
                    self._finish_cancellation(job.id)
                )
            else:
                self.log.info("reattached logical allocation for running job %s", job.id)

    async def _reap_recovered_jobs_locked(self) -> None:
        current_boot = boot_id()
        for job in self.db.active_jobs():
            if job.id in self._processes:
                continue
            if job.boot_id and current_boot and job.boot_id != current_boot:
                reason = "interrupted by machine reboot"
            elif managed_process_group_alive(job.pid, job.pgid, job.process_start_time):
                continue
            else:
                reason = "process exited after daemon restart; exit status unavailable"
            state = JobState.CANCELLED if job.state is JobState.CANCELLING else JobState.FAILED
            self.db.finish(job.id, state, reason=reason)
            self.log.warning("reaped recovered job %s as %s", job.id, state.value)
