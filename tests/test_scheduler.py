from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from conftest import FakeGPUProvider

from gq.database import Database
from gq.models import ComputeProcess, GPUObservation, GPUState, JobState
from gq.scheduler import DuplicateKey, Scheduler, SchedulerError


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


# --- groups, keys, bulk cancel, retry, and holding failing groups ------------------


async def wait_until(predicate, timeout: float = 5) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.03)
    raise AssertionError("condition was not met in time")


def gpus(count: int) -> FakeGPUProvider:
    return FakeGPUProvider([GPUObservation(f"GPU-{i}", i, "GPU") for i in range(count)])


async def states_of(scheduler: Scheduler, ids: list[int]) -> list[JobState]:
    result = []
    for job_id in ids:
        job = await scheduler.get_job(job_id)
        assert job is not None
        result.append(job.state)
    return result


@pytest.mark.asyncio
async def test_duplicate_key_is_refused_until_the_holder_fails(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    holder = await scheduler.submit(["sleep", "30"], tmp_path, {}, 1, key="cell-a")
    await wait_for_state(scheduler, holder.id, {JobState.RUNNING})
    with pytest.raises(DuplicateKey) as blocked:
        await scheduler.submit(["sleep", "30"], tmp_path, {}, 1, key="cell-a")
    assert blocked.value.existing.id == holder.id

    await scheduler.cancel(holder.id)
    await wait_for_state(scheduler, holder.id, {JobState.CANCELLED})
    again = await scheduler.submit(["true"], tmp_path, {}, 1, key="cell-a")
    await wait_for_state(scheduler, again.id, {JobState.DONE})
    with pytest.raises(DuplicateKey, match="already DONE"):
        await scheduler.submit(["true"], tmp_path, {}, 1, key="cell-a")
    await scheduler.close()


@pytest.mark.asyncio
async def test_labels_are_validated(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    for bad in ("", "   ", "x" * 201, "line\nbreak"):
        with pytest.raises(SchedulerError):
            await scheduler.submit(["true"], tmp_path, {}, 1, group=bad)
    await scheduler.close()


@pytest.mark.asyncio
async def test_cancel_many_handles_every_state_in_one_pass(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    running = await scheduler.submit(["sleep", "30"], tmp_path, {}, 1)
    await wait_for_state(scheduler, running.id, {JobState.RUNNING})
    waiting = await scheduler.submit(["sleep", "30"], tmp_path, {}, 1)
    done_job = await scheduler.submit(["true"], tmp_path, {}, 1)
    await scheduler.cancel(done_job.id)  # still waiting behind the running job

    result = await scheduler.cancel_many([running.id, waiting.id, done_job.id, 999])
    assert result == {
        "cancelled": [waiting.id],
        "cancelling": [running.id],
        "skipped": [done_job.id, 999],
    }
    await wait_for_state(scheduler, running.id, {JobState.CANCELLED})
    # The freed GPU must not have launched the job this batch cancelled.
    assert await states_of(scheduler, [waiting.id]) == [JobState.CANCELLED]
    await scheduler.close()


@pytest.mark.asyncio
async def test_retry_resubmits_with_original_context_exactly_once(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    failed = await scheduler.submit(
        ["false"], tmp_path, {"SEED": "3"}, 1, name="cell", group="sweep", key="k"
    )
    await wait_for_state(scheduler, failed.id, {JobState.FAILED})

    first = await scheduler.retry([failed.id])
    assert first["skipped"] == []
    [created] = first["created"]  # type: ignore[misc]
    attempt = await scheduler.get_job(created["job_id"])
    assert attempt is not None
    assert (attempt.argv, attempt.env, attempt.name) == (["false"], {"SEED": "3"}, "cell")
    assert (attempt.group, attempt.key, attempt.retry_of) == ("sweep", "k", failed.id)

    # Running the same retry again must not queue a duplicate.
    second = await scheduler.retry([failed.id])
    assert second["created"] == []
    assert "already retried" in str(second["skipped"])

    await wait_for_state(scheduler, attempt.id, {JobState.FAILED})
    third = await scheduler.retry([attempt.id])
    [chained] = third["created"]  # type: ignore[misc]
    chained_job = await scheduler.get_job(chained["job_id"])
    assert chained_job is not None and chained_job.retry_of == failed.id  # lineage root
    await wait_for_state(scheduler, chained_job.id, {JobState.FAILED})
    await scheduler.close()


@pytest.mark.asyncio
async def test_retry_skips_jobs_that_did_not_fail(tmp_path: Path) -> None:
    scheduler = make_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    done = await scheduler.submit(["true"], tmp_path, {}, 1)
    await wait_for_state(scheduler, done.id, {JobState.DONE})
    result = await scheduler.retry([done.id, 12345])
    assert result["created"] == []
    reasons = str(result["skipped"])
    assert "is DONE" in reasons and "does not exist" in reasons
    await scheduler.close()


def fail_fast_scheduler(tmp_path: Path, provider: FakeGPUProvider, count: int = 3) -> Scheduler:
    return Scheduler(
        Database(tmp_path / "gq.sqlite3"),
        provider,
        tmp_path / "logs",
        cancel_grace_seconds=0.2,
        fail_fast_count=count,
        fail_fast_seconds=60,
    )


@pytest.mark.asyncio
async def test_group_is_held_after_consecutive_quick_failures(tmp_path: Path) -> None:
    scheduler = fail_fast_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    jobs = [await scheduler.submit(["false"], tmp_path, {}, 1, group="sweep") for _ in range(5)]
    ids = [job.id for job in jobs]

    async def held() -> bool:
        return "sweep" in scheduler.held_groups

    await wait_until(held)
    await asyncio.sleep(0.3)  # give a wrongly-launched job time to show up
    assert await states_of(scheduler, ids) == [JobState.FAILED] * 3 + [JobState.WAITING] * 2
    assert "3 jobs in a row failed" in scheduler.held_groups["sweep"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_launch_failures_in_one_pass_stop_at_the_threshold(tmp_path: Path) -> None:
    """A mistyped executable fails synchronously inside the scheduling loop itself."""
    scheduler = fail_fast_scheduler(tmp_path, gpus(4))
    await scheduler.initialize()
    await scheduler.set_paused(True)  # queue everything, then release it in one pass
    ids = [
        (await scheduler.submit(["/nonexistent/trainer"], tmp_path, {}, 1, group="typo")).id
        for _ in range(20)
    ]
    await scheduler.set_paused(False)

    states = await states_of(scheduler, ids)
    assert states.count(JobState.FAILED) == 3
    assert states.count(JobState.WAITING) == 17
    assert "typo" in scheduler.held_groups
    await scheduler.close()


@pytest.mark.asyncio
async def test_a_success_breaks_the_failure_streak(tmp_path: Path) -> None:
    scheduler = fail_fast_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    commands = [["false"], ["false"], ["true"], ["false"], ["false"]]
    ids = [(await scheduler.submit(c, tmp_path, {}, 1, group="mixed")).id for c in commands]
    await wait_for_state(scheduler, ids[-1], {JobState.FAILED})
    assert "mixed" not in scheduler.held_groups
    await scheduler.close()


@pytest.mark.asyncio
async def test_ungrouped_jobs_and_disabled_fail_fast_are_never_held(tmp_path: Path) -> None:
    scheduler = fail_fast_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    ids = [(await scheduler.submit(["false"], tmp_path, {}, 1)).id for _ in range(4)]
    await wait_for_state(scheduler, ids[-1], {JobState.FAILED})
    assert scheduler.held_groups == {}
    await scheduler.close()

    disabled = fail_fast_scheduler(tmp_path / "off", gpus(1), count=0)
    await disabled.initialize()
    grouped = [(await disabled.submit(["false"], tmp_path, {}, 1, group="g")).id for _ in range(4)]
    await wait_for_state(disabled, grouped[-1], {JobState.FAILED})
    assert disabled.held_groups == {}
    await disabled.close()


@pytest.mark.asyncio
async def test_releasing_a_hold_only_counts_failures_after_the_release(tmp_path: Path) -> None:
    scheduler = fail_fast_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    ids = [(await scheduler.submit(["false"], tmp_path, {}, 1, group="g")).id for _ in range(5)]

    async def held() -> bool:
        return "g" in scheduler.held_groups

    await wait_until(held)
    assert await scheduler.set_group_held("g", False) is True
    # The two remaining jobs now run and fail, but two is below the threshold once the
    # three failures from before the release stop counting.
    await wait_for_state(scheduler, ids[-1], {JobState.FAILED})
    assert "g" not in scheduler.held_groups
    await scheduler.close()


@pytest.mark.asyncio
async def test_retry_releases_the_hold_on_its_group(tmp_path: Path) -> None:
    scheduler = fail_fast_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    ids = [(await scheduler.submit(["false"], tmp_path, {}, 1, group="g")).id for _ in range(4)]

    async def held() -> bool:
        return "g" in scheduler.held_groups

    await wait_until(held)
    result = await scheduler.retry([ids[0]])
    assert result["released_groups"] == ["g"]
    assert "g" not in scheduler.held_groups
    await wait_for_state(scheduler, ids[3], {JobState.FAILED})  # the held job ran
    await scheduler.close()


@pytest.mark.asyncio
async def test_holds_survive_a_daemon_restart(tmp_path: Path) -> None:
    first = fail_fast_scheduler(tmp_path, gpus(1))
    await first.initialize()
    ids = [(await first.submit(["false"], tmp_path, {}, 1, group="g")).id for _ in range(4)]

    async def held() -> bool:
        return "g" in first.held_groups

    await wait_until(held)
    await first.close()
    first.db.close()

    second = fail_fast_scheduler(tmp_path, gpus(1))
    await second.initialize()
    await second.tick()
    assert "g" in second.held_groups
    assert await states_of(second, [ids[3]]) == [JobState.WAITING]
    summaries = await second.group_summaries()
    assert summaries == [
        {
            "group": "g",
            "counts": {"FAILED": 3, "WAITING": 1},
            "retried": 0,
            "held": True,
            "hold_reason": second.held_groups["g"],
        }
    ]
    await second.close()


@pytest.mark.asyncio
async def test_group_counts_follow_each_cells_latest_attempt(tmp_path: Path) -> None:
    """A cell that failed and then succeeded on retry is DONE, not DONE plus FAILED."""
    scheduler = make_scheduler(tmp_path, gpus(1))
    await scheduler.initialize()
    marker = tmp_path / "fixed"
    command = ["sh", "-c", f"test -f {marker}"]
    failed = await scheduler.submit(command, tmp_path, {}, 1, group="g")
    await wait_for_state(scheduler, failed.id, {JobState.FAILED})
    marker.touch()
    [created] = (await scheduler.retry([failed.id]))["created"]  # type: ignore[misc]
    await wait_for_state(scheduler, created["job_id"], {JobState.DONE})

    [summary] = await scheduler.group_summaries()
    assert summary["counts"] == {"DONE": 1}
    assert summary["retried"] == 1
    await scheduler.close()
