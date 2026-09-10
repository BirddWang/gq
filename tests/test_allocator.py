from __future__ import annotations

from pathlib import Path

from gq.allocator import classify_gpus, select_gpus
from gq.models import (
    ComputeProcess,
    GPUObservation,
    GPUState,
    Job,
    JobState,
)


def job(state: JobState = JobState.RUNNING) -> Job:
    return Job(
        id=7,
        argv=["python", "train.py"],
        cwd=Path("/tmp"),
        env={},
        requested_gpus=1,
        state=state,
        submit_time="now",
        pgid=100,
    )


def test_unknown_nvml_pid_is_external() -> None:
    observations = [
        GPUObservation("A", 0, "GPU", compute_processes=(ComputeProcess(123, 100, "python"),))
    ]
    status = classify_gpus(observations, {}, {}, lambda _pid, _job: False)
    assert status[0].state is GPUState.EXTERNAL


def test_external_becomes_free_when_pid_disappears() -> None:
    occupied = GPUObservation("A", 0, "GPU", compute_processes=(ComputeProcess(123),))
    empty = GPUObservation("A", 0, "GPU")
    assert classify_gpus([occupied], {}, {}, lambda *_: False)[0].state is GPUState.EXTERNAL
    assert classify_gpus([empty], {}, {}, lambda *_: False)[0].state is GPUState.FREE


def test_starting_reservation_remains_reserved_before_cuda_appears() -> None:
    managed = job(JobState.STARTING)
    status = classify_gpus([GPUObservation("A", 0, "GPU")], {"A": 7}, {7: managed}, lambda *_: True)
    assert status[0].state is GPUState.RESERVED
    assert select_gpus(status, 1) == []


def test_conflicting_pid_on_managed_gpu_fails_closed() -> None:
    managed = job()
    observation = GPUObservation(
        "A",
        0,
        "GPU",
        compute_processes=(ComputeProcess(101), ComputeProcess(202)),
    )
    status = classify_gpus([observation], {"A": 7}, {7: managed}, lambda pid, _: pid == 101)[0]
    assert status.state is GPUState.UNKNOWN
    assert "202" in (status.detail or "")


def test_lowest_free_indices_selected_deterministically() -> None:
    observations = [
        GPUObservation("C", 3, "GPU"),
        GPUObservation("A", 0, "GPU"),
        GPUObservation("B", 2, "GPU"),
    ]
    statuses = classify_gpus(observations, {}, {}, lambda *_: False)
    assert [gpu.index for gpu in select_gpus(statuses, 2)] == [0, 2]
