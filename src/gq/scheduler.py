from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from pathlib import Path

from .allocator import classify_gpus, select_gpus
from .database import Database
from .models import GPUObservation, GPUState, GPUStatus, Job, JobState
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


class Scheduler:
    """The daemon's sole mutation context for jobs and GPU allocations."""

    def __init__(
        self,
        database: Database,
        gpu_provider: GPUProvider,
        logs_dir: Path,
        *,
        cancel_grace_seconds: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db = database
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

    async def initialize(self) -> None:
        async with self.lock:
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
    ) -> Job:
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
            placeholder = self.logs_dir / "pending.log"
            job = self.db.create_job(
                argv,
                cwd.resolve(),
                env,
                requested_gpus,
                name or Path(argv[0]).name,
                placeholder,
            )
            log_path = self.logs_dir / f"{job.id}.log"
            self.db.set_log_path(job.id, log_path)
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

    async def list_jobs(self, limit: int | None = None) -> list[Job]:
        async with self.lock:
            return self.db.list_jobs(limit)

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
            if job.state is JobState.WAITING:
                self.db.finish(job.id, JobState.CANCELLED, reason="cancelled before launch")
                self.log.info("cancelled waiting job %s", job.id)
                await self._try_schedule_locked()
                return "cancelled"
            if job.state.terminal:
                raise SchedulerError(f"job {job_id} is already {job.state.value}")
            if job.state is JobState.CANCELLING:
                return "cancellation already in progress"
            if not self.db.set_cancelling(job.id):
                raise SchedulerError(f"job {job_id} cannot be cancelled from {job.state.value}")
            sent = signal_managed_group(job.pid, job.pgid, job.process_start_time, signal.SIGTERM)
            self.log.info("cancelling job %s with SIGTERM (sent=%s)", job.id, sent)
            task = asyncio.create_task(self._finish_cancellation(job.id))
            self._cancellation_tasks[job.id] = task
            return "cancellation requested"

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
        if self._stopping:
            return
        statuses = self._gpu_status_locked()
        for waiting in self.db.waiting_jobs():
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
                else:
                    state = JobState.DONE if exit_code == 0 else JobState.FAILED
                    reason = None if exit_code == 0 else f"command exited with status {exit_code}"
                    self.db.finish(job.id, state, exit_code=exit_code, reason=reason)
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
