from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gq.database import BASELINE, SCHEMA_VERSION, Database, KeyInUse, SchemaTooNew, utc_now
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


def _legacy_database(path: Path, tmp_path: Path) -> None:
    """Build a database exactly as gq 0.1.0 left it: baseline tables, user_version 0."""
    connection = sqlite3.connect(path, isolation_level=None)
    for statement in BASELINE:
        connection.execute(statement)
    connection.execute(
        """INSERT INTO jobs (id, name, state, command_json, cwd, env_json, requested_gpus,
           submit_time, start_time, pid, pgid, log_path)
           VALUES (7, 'legacy', 'RUNNING', '["true"]', ?, '{"A": "B"}', 1,
                   '2026-08-31T00:00:00+00:00', '2026-08-31T00:00:01+00:00', 1, 1, ?)""",
        (str(tmp_path), str(tmp_path / "7.log")),
    )
    connection.execute("INSERT INTO job_gpus VALUES (7, 'GPU-A', 0)")
    connection.close()


def test_pre_versioning_database_is_migrated_without_losing_jobs(tmp_path: Path) -> None:
    """The full v0 -> v1 -> current path, against the real 0.1.0 schema."""
    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path, tmp_path)

    db = Database(path)
    assert db.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    survivor = db.get_job(7)
    assert survivor is not None
    assert survivor.state is JobState.RUNNING
    assert survivor.env == {"A": "B"}
    assert survivor.gpu_uuids == ["GPU-A"]
    assert (survivor.group, survivor.key, survivor.retry_of) == (None, None, None)
    # A migrated database must still be writable through the new columns.
    fresh = db.create_job(["true"], tmp_path, {}, 1, None, tmp_path / "8.log", group="g", key="k")
    assert (fresh.group, fresh.key) == ("g", "k")
    db.close()


def test_reopening_a_current_database_does_not_rerun_migrations(tmp_path: Path) -> None:
    path = tmp_path / "gq.sqlite3"
    Database(path).close()
    Database(path).close()  # would raise "duplicate column" if migration 2 ran twice


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


def test_key_is_taken_until_its_holder_fails_or_is_cancelled(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    first = db.create_job(["a"], tmp_path, {}, 1, None, tmp_path / "1.log", key="cell")
    with pytest.raises(KeyInUse) as waiting:
        db.create_job(["a"], tmp_path, {}, 1, None, tmp_path / "2.log", key="cell")
    assert (waiting.value.job_id, waiting.value.state) == (first.id, JobState.WAITING)

    db.finish(first.id, JobState.DONE, exit_code=0)
    with pytest.raises(KeyInUse, match="already DONE"):
        db.create_job(["a"], tmp_path, {}, 1, None, tmp_path / "2.log", key="cell")

    # Only a failed or cancelled attempt frees the key.
    other = db.create_job(["b"], tmp_path, {}, 1, None, tmp_path / "3.log", key="other")
    db.finish(other.id, JobState.FAILED, exit_code=1)
    again = db.create_job(["b"], tmp_path, {}, 1, None, tmp_path / "4.log", key="other")
    db.finish(again.id, JobState.CANCELLED)
    db.create_job(["b"], tmp_path, {}, 1, None, tmp_path / "5.log", key="other")
    # Jobs without a key never conflict.
    db.create_job(["c"], tmp_path, {}, 1, None, tmp_path / "6.log")
    db.create_job(["c"], tmp_path, {}, 1, None, tmp_path / "7.log")
    db.close()


def test_list_jobs_filters_combine(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    ids = [
        db.create_job(["x"], tmp_path, {}, 1, None, tmp_path / f"{i}.log", group=group).id
        for i, group in enumerate(["a", "a", "b", "a", None])
    ]
    db.finish(ids[1], JobState.FAILED, exit_code=1)

    def listed(**filters: object) -> list[int]:
        return [job.id for job in db.list_jobs(**filters)]  # type: ignore[arg-type]

    assert listed(group="a") == [ids[3], ids[1], ids[0]]
    assert listed(group="a", states=[JobState.WAITING]) == [ids[3], ids[0]]
    assert listed(id_ranges=[(ids[0], ids[1]), (ids[4], ids[4])]) == [ids[4], ids[1], ids[0]]
    assert listed(group="a", id_ranges=[(ids[1], ids[4])]) == [ids[3], ids[1]]
    assert listed(limit=2) == [ids[4], ids[3]]
    db.close()


def test_newer_attempt_follows_the_retry_lineage(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    root = db.create_job(["x"], tmp_path, {}, 1, None, tmp_path / "1.log")
    unrelated = db.create_job(["y"], tmp_path, {}, 1, None, tmp_path / "2.log")
    assert db.newer_attempt(root.id, root.id) is None
    retry = db.create_job(["x"], tmp_path, {}, 1, None, tmp_path / "3.log", retry_of=root.id)
    assert db.newer_attempt(root.id, root.id) == retry.id
    assert db.newer_attempt(root.id, retry.id) is None
    assert db.newer_attempt(unrelated.id, unrelated.id) is None
    db.close()


def test_group_holds_and_release_time(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    assert db.set_group_hold("sweep", False) is False  # releasing an unheld group is a no-op
    assert db.set_group_hold("sweep", True, "3 quick failures") is True
    assert db.set_group_hold("sweep", True, "again") is False
    assert db.held_groups() == {"sweep": "3 quick failures"}
    assert db.group_released_at("sweep") is None
    assert db.set_group_hold("sweep", False) is True
    assert db.held_groups() == {}
    assert db.group_released_at("sweep") is not None
    db.close()


def test_recent_group_outcomes_ignore_cancellations_and_older_history(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite3")

    def ended(state: JobState, when: str) -> int:
        job = db.create_job(["x"], tmp_path, {}, 1, None, tmp_path / "x.log", group="g")
        db.finish(job.id, state, exit_code=0 if state is JobState.DONE else 1)
        db.connection.execute("UPDATE jobs SET end_time = ? WHERE id = ?", (when, job.id))
        return job.id

    old_fail = ended(JobState.FAILED, "2026-01-01T00:00:00+00:00")
    done = ended(JobState.DONE, "2026-01-02T00:00:00+00:00")
    ended(JobState.CANCELLED, "2026-01-03T00:00:00+00:00")
    new_fail = ended(JobState.FAILED, "2026-01-04T00:00:00+00:00")

    assert [j.id for j in db.recent_group_outcomes("g", None, 10)] == [new_fail, done, old_fail]
    since = "2026-01-01T12:00:00+00:00"
    assert [j.id for j in db.recent_group_outcomes("g", since, 10)] == [new_fail, done]
    assert [j.id for j in db.recent_group_outcomes("g", None, 1)] == [new_fail]
    db.close()
