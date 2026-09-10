from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest
from conftest import FakeGPUProvider

from gq.database import Database
from gq.models import GPUObservation, JobState
from gq.processes import boot_id, process_start_time
from gq.scheduler import Scheduler


def persisted_running_job(db: Database, tmp_path: Path, process: subprocess.Popen[bytes]):
    job = db.create_job(["sleep", "30"], tmp_path, dict(os.environ), 1, None, tmp_path / "log")
    db.reserve(job.id, [GPUObservation("A", 0, "GPU")])
    db.mark_running(
        job.id,
        process.pid,
        os.getpgid(process.pid),
        process_start_time(process.pid),
        boot_id(),
    )
    return db.get_job(job.id)


@pytest.mark.asyncio
async def test_restart_reattaches_live_process_then_reaps_it(tmp_path: Path) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    db = Database(tmp_path / "db.sqlite3")
    job = persisted_running_job(db, tmp_path, process)
    assert job is not None
    scheduler = Scheduler(db, FakeGPUProvider([GPUObservation("A", 0, "GPU")]), tmp_path / "logs")
    await scheduler.initialize()
    assert (await scheduler.get_job(job.id)).state is JobState.RUNNING  # type: ignore[union-attr]
    os.killpg(process.pid, signal.SIGTERM)
    process.wait(timeout=2)
    await scheduler.tick()
    recovered = await scheduler.get_job(job.id)
    assert recovered and recovered.state is JobState.FAILED
    assert "exit status unavailable" in (recovered.failure_reason or "")
    await scheduler.close()


@pytest.mark.asyncio
async def test_dead_saved_pid_fails_during_startup(tmp_path: Path) -> None:
    process = subprocess.Popen(["/bin/true"], start_new_session=True)
    process.wait()
    db = Database(tmp_path / "db.sqlite3")
    job = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "log")
    db.reserve(job.id, [GPUObservation("A", 0, "GPU")])
    db.mark_running(job.id, process.pid, process.pid, 1, boot_id())
    scheduler = Scheduler(db, FakeGPUProvider([GPUObservation("A", 0, "GPU")]), tmp_path / "logs")
    await scheduler.initialize()
    recovered = await scheduler.get_job(job.id)
    assert recovered and recovered.state is JobState.FAILED
    assert recovered.gpu_uuids == ["A"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_changed_boot_id_marks_job_interrupted(tmp_path: Path) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        db = Database(tmp_path / "db.sqlite3")
        job = db.create_job(["sleep", "30"], tmp_path, {}, 1, None, tmp_path / "log")
        db.reserve(job.id, [GPUObservation("A", 0, "GPU")])
        db.mark_running(
            job.id, process.pid, process.pid, process_start_time(process.pid), "old-boot"
        )
        scheduler = Scheduler(
            db, FakeGPUProvider([GPUObservation("A", 0, "GPU")]), tmp_path / "logs"
        )
        await scheduler.initialize()
        recovered = await scheduler.get_job(job.id)
        assert recovered and recovered.state is JobState.FAILED
        assert recovered.failure_reason == "interrupted by machine reboot"
        await scheduler.close()
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
