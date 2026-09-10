from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from .models import GPUObservation, GPUState, GPUStatus, Job, JobState


def classify_gpus(
    observations: Iterable[GPUObservation],
    assignments: Mapping[str, int],
    active_jobs: Mapping[int, Job],
    pid_is_managed: Callable[[int, Job], bool],
) -> list[GPUStatus]:
    """Merge logical reservations with physical observations.

    A reservation always prevents allocation, including before CUDA appears. Any
    unrecognized compute PID also prevents allocation, regardless of utilization.
    """
    statuses: list[GPUStatus] = []
    for observation in sorted(observations, key=lambda item: item.index):
        job_id = assignments.get(observation.uuid)
        job = active_jobs.get(job_id) if job_id is not None else None
        if observation.error:
            state = GPUState.UNKNOWN
            detail = observation.error
        elif job_id is not None and job is None:
            state = GPUState.UNKNOWN
            detail = f"stale reservation for job {job_id}"
        elif job is not None:
            conflicts = [
                proc.pid
                for proc in observation.compute_processes
                if not pid_is_managed(proc.pid, job)
            ]
            if job.state is JobState.ORPHANED:
                state = GPUState.UNKNOWN
                detail = f"orphaned allocation for job {job.id}"
            elif conflicts:
                state = GPUState.UNKNOWN
                detail = "conflicting external process(es): " + ", ".join(map(str, conflicts))
            else:
                state = GPUState.RESERVED if job.state is JobState.STARTING else GPUState.RUNNING
                detail = None
        elif observation.compute_processes:
            state = GPUState.EXTERNAL
            detail = None
        else:
            state = GPUState.FREE
            detail = None
        statuses.append(
            GPUStatus(
                uuid=observation.uuid,
                index=observation.index,
                name=observation.name,
                state=state,
                owner_job_id=job_id,
                processes=observation.compute_processes,
                total_memory=observation.total_memory,
                used_memory=observation.used_memory,
                detail=detail,
            )
        )
    return statuses


def select_gpus(statuses: Iterable[GPUStatus], count: int) -> list[GPUStatus]:
    free = sorted(
        (status for status in statuses if status.state is GPUState.FREE),
        key=lambda status: status.index,
    )
    return free[:count] if len(free) >= count else []
