from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from conftest import FakeGPUProvider

from gq.database import Database
from gq.models import ComputeProcess, GPUObservation, GPUState, JobState
from gq.scheduler import Scheduler, SchedulerError


async def wait_for_state(
    scheduler: Scheduler,
    job_id: int,
    states: set[JobState],
    timeout: float = 5,
) -> JobState:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        job = await scheduler.get_job(job_id)
        assert job is not None
        if job.state in states:
            return job.state
        await asyncio.sleep(0.03)
    raise AssertionError(f"job {job_id} did not reach {states}")


def make_scheduler(tmp_path: Path, provider: FakeGPUProvider, grace: float = 0.2) -> Scheduler:
    return Scheduler(
        Database(tmp_path / "gq.sqlite3"),
        provider,
        tmp_path / "logs",
        cancel_grace_seconds=grace,
    )


@pytest.mark.asyncio
async def test_one_free_gpu_runs_and_sets_cuda_visible_devices(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    output = tmp_path / "visible.txt"
    job = await scheduler.submit(
        ["/bin/sh", "-c", f'printf %s "$CUDA_VISIBLE_DEVICES" > {output}'],
        tmp_path,
        dict(os.environ),
        1,
    )
    assert await wait_for_state(scheduler, job.id, {JobState.DONE}) is JobState.DONE
    assert output.read_text() == "0"
    await scheduler.close()


@pytest.mark.asyncio
async def test_external_gpu_causes_job_to_wait_then_run(tmp_path: Path) -> None:
    provider = FakeGPUProvider(
        [GPUObservation("A", 0, "GPU", compute_processes=(ComputeProcess(999999),))]
    )
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    job = await scheduler.submit(["/bin/true"], tmp_path, dict(os.environ), 1)
    assert (await scheduler.get_job(job.id)).state is JobState.WAITING  # type: ignore[union-attr]
    provider.observations = [GPUObservation("A", 0, "GPU")]
    await scheduler.tick()
    assert await wait_for_state(scheduler, job.id, {JobState.DONE}) is JobState.DONE
    await scheduler.close()


@pytest.mark.asyncio
async def test_simple_backfilling_skips_large_blocked_job(tmp_path: Path) -> None:
    provider = FakeGPUProvider(
        [
            GPUObservation("A", 0, "GPU"),
            GPUObservation("B", 1, "GPU"),
            GPUObservation("C", 2, "GPU"),
            GPUObservation("D", 3, "GPU", compute_processes=(ComputeProcess(999999),)),
        ]
    )
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    large = await scheduler.submit(["/bin/true"], tmp_path, dict(os.environ), 4, "large")
    small = await scheduler.submit(
        ["/bin/sh", "-c", "sleep 0.2"], tmp_path, dict(os.environ), 1, "small"
    )
    assert (await scheduler.get_job(large.id)).state is JobState.WAITING  # type: ignore[union-attr]
    assert (await scheduler.get_job(small.id)).state in {JobState.RUNNING, JobState.DONE}  # type: ignore[union-attr]
    await wait_for_state(scheduler, small.id, {JobState.DONE})
    await scheduler.cancel(large.id)
    await scheduler.close()


@pytest.mark.asyncio
async def test_two_allocations_never_overlap(tmp_path: Path, gpu_factory) -> None:
    provider = FakeGPUProvider(gpu_factory(4))
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    first, second = await asyncio.gather(
        scheduler.submit(["/bin/sh", "-c", "sleep 10"], tmp_path, dict(os.environ), 2),
        scheduler.submit(["/bin/sh", "-c", "sleep 10"], tmp_path, dict(os.environ), 2),
    )
    first_job = await scheduler.get_job(first.id)
    second_job = await scheduler.get_job(second.id)
    assert first_job and second_job
    assert set(first_job.gpu_uuids).isdisjoint(second_job.gpu_uuids)
    assert len(set(first_job.gpu_uuids + second_job.gpu_uuids)) == 4
    await scheduler.cancel(first.id)
    await scheduler.cancel(second.id)
    await wait_for_state(scheduler, first.id, {JobState.CANCELLED})
    await wait_for_state(scheduler, second.id, {JobState.CANCELLED})
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancel_terminates_entire_process_group(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    child_file = tmp_path / "child.pid"
    command = f"sleep 30 & child=$!; echo $child > {child_file}; wait $child"
    job = await scheduler.submit(["/bin/sh", "-c", command], tmp_path, dict(os.environ), 1)
    deadline = asyncio.get_running_loop().time() + 2
    while not child_file.exists() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    child_pid = int(child_file.read_text())
    await scheduler.cancel(job.id)
    assert await wait_for_state(scheduler, job.id, {JobState.CANCELLED}) is JobState.CANCELLED
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    await scheduler.close()


@pytest.mark.asyncio
async def test_nonzero_exit_fails_and_releases_reservation(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    job = await scheduler.submit(["/bin/sh", "-c", "exit 17"], tmp_path, dict(os.environ), 1)
    assert await wait_for_state(scheduler, job.id, {JobState.FAILED}) is JobState.FAILED
    final = await scheduler.get_job(job.id)
    assert final and final.exit_code == 17 and final.gpu_uuids == ["A"]
    assert (await scheduler.gpu_status())[0].state is GPUState.FREE
    await scheduler.close()


@pytest.mark.asyncio
async def test_rejects_impossible_gpu_request(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, FakeGPUProvider([GPUObservation("A", 0, "GPU")]))
    await scheduler.initialize()
    with pytest.raises(SchedulerError, match="machine has 1"):
        await scheduler.submit(["true"], tmp_path, dict(os.environ), 2)
    assert await scheduler.list_jobs() == []
    await scheduler.close()


@pytest.mark.asyncio
async def test_deleting_a_job_removes_its_log(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    job = await scheduler.submit(["true"], tmp_path, {}, 1)
    await wait_for_state(scheduler, job.id, {JobState.DONE})

    log_path = tmp_path / "logs" / f"{job.id}.log"
    assert log_path.exists()
    removed = await scheduler.delete_jobs([job.id])

    assert [item.id for item in removed] == [job.id]
    assert not log_path.exists()
    assert await scheduler.get_job(job.id) is None
    await scheduler.close()


@pytest.mark.asyncio
async def test_deleting_a_running_job_is_refused(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    job = await scheduler.submit(["sleep", "30"], tmp_path, {}, 1)
    await wait_for_state(scheduler, job.id, {JobState.RUNNING})

    with pytest.raises(SchedulerError, match="cannot be deleted"):
        await scheduler.delete_jobs([job.id])
    assert await scheduler.get_job(job.id) is not None

    await scheduler.cancel(job.id)
    await wait_for_state(scheduler, job.id, {JobState.CANCELLED})
    await scheduler.close()


@pytest.mark.asyncio
async def test_cleanup_never_unlinks_outside_the_log_directory(tmp_path: Path) -> None:
    """A corrupted or hand-edited log_path must not make cleanup delete arbitrary files."""
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    job = await scheduler.submit(["true"], tmp_path, {}, 1)
    await wait_for_state(scheduler, job.id, {JobState.DONE})

    bystander = tmp_path / "important.txt"
    bystander.write_text("do not delete me")
    scheduler.db.set_log_path(job.id, bystander)

    await scheduler.delete_jobs([job.id])

    assert bystander.read_text() == "do not delete me"
    await scheduler.close()


@pytest.mark.asyncio
async def test_paused_queue_accepts_jobs_but_does_not_start_them(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    await scheduler.set_paused(True)

    job = await scheduler.submit(["true"], tmp_path, {}, 1)
    await scheduler.tick()
    current = await scheduler.get_job(job.id)
    assert current is not None and current.state is JobState.WAITING

    await scheduler.set_paused(False)
    await wait_for_state(scheduler, job.id, {JobState.DONE})
    await scheduler.close()


@pytest.mark.asyncio
async def test_pausing_does_not_touch_running_jobs(tmp_path: Path) -> None:
    provider = FakeGPUProvider([GPUObservation("A", 0, "GPU")])
    scheduler = make_scheduler(tmp_path, provider)
    await scheduler.initialize()
    running = await scheduler.submit(["sleep", "0.5"], tmp_path, {}, 1)
    await wait_for_state(scheduler, running.id, {JobState.RUNNING})
    queued = await scheduler.submit(["true"], tmp_path, {}, 1)

    await scheduler.set_paused(True)
    assert [job.id for job in await scheduler.active_jobs()] == [running.id]
    # The running job still finishes normally, and its freed GPU is not handed on.
    await wait_for_state(scheduler, running.id, {JobState.DONE})
    await scheduler.tick()
    current = await scheduler.get_job(queued.id)
    assert current is not None and current.state is JobState.WAITING
    await scheduler.close()
