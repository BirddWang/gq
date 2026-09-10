from __future__ import annotations

from pathlib import Path

import pytest

from gq.database import SCHEMA_VERSION, Database, SchemaTooNew, utc_now
from gq.models import GPUObservation, JobState


def test_job_ids_state_and_assignments_are_persistent(tmp_path: Path) -> None:
    path = tmp_path / "gq.sqlite3"
    db = Database(path)
    first = db.create_job(["true"], tmp_path, {"A": "B"}, 1, "one", tmp_path / "1.log")
    second = db.create_job(["false"], tmp_path, {}, 1, None, tmp_path / "2.log")
    assert (first.id, second.id) == (1, 2)
    db.reserve(first.id, [GPUObservation("GPU-A", 2, "Fake")])
    db.mark_running(first.id, 123, 123, 456, "boot")
    assert db.assignments() == {"GPU-A": first.id}
    db.close()

    reopened = Database(path)
    recovered = reopened.get_job(first.id)
    assert recovered is not None
    assert recovered.state is JobState.RUNNING
    assert recovered.env == {"A": "B"}
    assert recovered.gpu_uuids == ["GPU-A"]
    reopened.finish(first.id, JobState.DONE, exit_code=0)
    assert reopened.assignments() == {}
    reopened.close()


def test_atomic_reservation_requires_waiting_state(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    job = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "log")
    gpu = GPUObservation("A", 0, "GPU")
    db.reserve(job.id, [gpu])
    try:
        db.reserve(job.id, [gpu])
    except RuntimeError as exc:
        assert "no longer waiting" in str(exc)
    else:
        raise AssertionError("second reservation unexpectedly succeeded")
    db.close()


def test_fresh_database_is_stamped_at_current_schema_version(tmp_path: Path) -> None:
    db = Database(tmp_path / "gq.sqlite3")
    version = db.connection.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    db.close()


def test_pre_versioning_database_is_adopted_without_losing_jobs(tmp_path: Path) -> None:
    """A 0.1.0 database has the tables but user_version 0. It must be stamped, not rebuilt."""
    path = tmp_path / "legacy.sqlite3"
    db = Database(path)
    job = db.create_job(["true"], tmp_path, {"A": "B"}, 1, "legacy", tmp_path / "1.log")
    db.reserve(job.id, [GPUObservation("GPU-A", 0, "Fake")])
    db.connection.execute("PRAGMA user_version = 0")
    db.close()

    reopened = Database(path)
    assert reopened.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    survivor = reopened.get_job(job.id)
    assert survivor is not None
    assert survivor.env == {"A": "B"}
    assert survivor.gpu_uuids == ["GPU-A"]
    reopened.close()


def test_database_from_a_newer_gq_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    db = Database(path)
    db.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    db.close()
    with pytest.raises(SchemaTooNew, match="upgrade gq"):
        Database(path)


def test_deleting_a_job_cascades_to_its_gpu_assignments(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    job = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "1.log")
    db.reserve(job.id, [GPUObservation("GPU-A", 0, "Fake")])
    db.mark_running(job.id, 1, 1, 2, "boot")
    db.finish(job.id, JobState.DONE, exit_code=0)

    assert db.delete_jobs([job.id])[0].id == job.id
    assert db.get_job(job.id) is None
    orphans = db.connection.execute("SELECT COUNT(*) FROM job_gpus").fetchone()[0]
    assert orphans == 0
    db.close()


def test_deleting_an_active_job_is_refused_and_rolls_back_the_batch(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    done = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "1.log")
    db.finish(done.id, JobState.DONE, exit_code=0)
    running = db.create_job(["sleep"], tmp_path, {}, 1, None, tmp_path / "2.log")
    db.reserve(running.id, [GPUObservation("GPU-A", 0, "Fake")])

    with pytest.raises(ValueError, match="cannot be deleted"):
        db.delete_jobs([done.id, running.id])
    # The terminal job listed first must survive the refusal.
    assert db.get_job(done.id) is not None
    db.close()


def test_terminal_jobs_before_filters_by_age_and_state(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    old = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "1.log")
    db.finish(old.id, JobState.FAILED, exit_code=1)
    db.connection.execute(
        "UPDATE jobs SET end_time = ? WHERE id = ?", ("2000-01-01T00:00:00+00:00", old.id)
    )
    recent = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "2.log")
    db.finish(recent.id, JobState.DONE, exit_code=0)
    waiting = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "3.log")

    cutoff = "2020-01-01T00:00:00+00:00"
    assert [job.id for job in db.terminal_jobs_before(cutoff)] == [old.id]
    assert db.terminal_jobs_before(cutoff, [JobState.DONE]) == []
    # A job that never ended has no end_time and must never be swept.
    assert waiting.id not in {job.id for job in db.terminal_jobs_before(utc_now())}
    db.close()
