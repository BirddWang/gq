from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import GPUObservation, Job, JobState


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA_VERSION = 2

BASELINE = [
    """CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    state TEXT NOT NULL,
    command_json TEXT NOT NULL,
    cwd TEXT NOT NULL,
    env_json TEXT NOT NULL,
    requested_gpus INTEGER NOT NULL CHECK (requested_gpus >= 1),
    submit_time TEXT NOT NULL,
    start_time TEXT,
    end_time TEXT,
    pid INTEGER,
    pgid INTEGER,
    process_start_time INTEGER,
    boot_id TEXT,
    exit_code INTEGER,
    log_path TEXT,
    failure_reason TEXT
)""",
    """CREATE TABLE IF NOT EXISTS job_gpus (
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    gpu_uuid TEXT NOT NULL,
    gpu_index_at_start INTEGER NOT NULL,
    PRIMARY KEY (job_id, gpu_uuid)
)""",
    "CREATE INDEX IF NOT EXISTS jobs_state_submit_idx ON jobs(state, submit_time, id)",
    """CREATE TABLE IF NOT EXISTS daemon_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""",
]

# Ordered upgrades keyed by the version they produce. Each list is applied in one
# transaction together with its `PRAGMA user_version` bump, so a partially applied
# migration cannot survive a crash.
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE jobs ADD COLUMN group_name TEXT",
        "ALTER TABLE jobs ADD COLUMN job_key TEXT",
        # Deliberately not a foreign key: deleting an old attempt with `gq rm` must not
        # be blocked by the attempts that retried it.
        "ALTER TABLE jobs ADD COLUMN retry_of INTEGER",
        "CREATE INDEX jobs_group_idx ON jobs(group_name, id)",
        "CREATE INDEX jobs_key_idx ON jobs(job_key)",
        "CREATE INDEX jobs_retry_idx ON jobs(retry_of)",
        """CREATE TABLE group_holds (
    group_name TEXT PRIMARY KEY,
    held INTEGER NOT NULL,
    reason TEXT,
    changed_at TEXT NOT NULL
)""",
    ],
}

# A key is taken while any job holding it could still succeed or already has. Only a
# failed or cancelled attempt frees it for another.
KEY_RELEASING_STATES = (JobState.FAILED.value, JobState.CANCELLED.value)


class SchemaTooNew(RuntimeError):
    pass


class KeyInUse(RuntimeError):
    def __init__(self, key: str, job_id: int, state: JobState) -> None:
        super().__init__(f"key {key!r} is already {state.value} as job {job_id}")
        self.key = key
        self.job_id = job_id
        self.state = state


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        # These three must run outside a transaction: journal_mode changes the file
        # format, and `foreign_keys` is a silent no-op mid-transaction. ON DELETE
        # CASCADE on job_gpus depends on it.
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    def migrate(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            # Either a fresh database or one created before versioning existed. The
            # baseline is idempotent, so the same path establishes both.
            with self.transaction() as conn:
                for statement in BASELINE:
                    conn.execute(statement)
                conn.execute("PRAGMA user_version = 1")
            version = 1
        if version > SCHEMA_VERSION:
            raise SchemaTooNew(
                f"database {self.path} is at schema version {version}, but this gq "
                f"understands at most {SCHEMA_VERSION}; upgrade gq"
            )
        while version < SCHEMA_VERSION:
            target = version + 1
            with self.transaction() as conn:
                for statement in MIGRATIONS[target]:
                    conn.execute(statement)
                # PRAGMA cannot be parameterized; target is a validated int.
                conn.execute(f"PRAGMA user_version = {target}")
            version = target

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def create_job(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        requested_gpus: int,
        name: str | None,
        log_path: Path,
        *,
        group: str | None = None,
        key: str | None = None,
        retry_of: int | None = None,
    ) -> Job:
        submitted = utc_now()
        with self.transaction() as conn:
            if key is not None:
                # Checked inside the insert's transaction, so two submissions of the
                # same key cannot both see it as free.
                holder = conn.execute(
                    f"""SELECT id, state FROM jobs WHERE job_key = ?
                        AND state NOT IN ({",".join("?" for _ in KEY_RELEASING_STATES)})
                        ORDER BY id DESC LIMIT 1""",
                    (key, *KEY_RELEASING_STATES),
                ).fetchone()
                if holder is not None:
                    raise KeyInUse(key, int(holder["id"]), JobState(holder["state"]))
            cursor = conn.execute(
                """INSERT INTO jobs
                   (name, state, command_json, cwd, env_json, requested_gpus,
                    submit_time, log_path, group_name, job_key, retry_of)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    JobState.WAITING.value,
                    json.dumps(argv),
                    str(cwd),
                    json.dumps(env),
                    requested_gpus,
                    submitted,
                    str(log_path),
                    group,
                    key,
                    retry_of,
                ),
            )
            row_id = cursor.lastrowid
            if row_id is None:  # SQLite always sets this after an INSERT.
                raise RuntimeError("SQLite reported no row id for the inserted job")
            job_id = row_id
        job = self.get_job(job_id)
        assert job is not None
        return job

    def set_log_path(self, job_id: int, log_path: Path) -> None:
        self.connection.execute(
            "UPDATE jobs SET log_path = ? WHERE id = ?", (str(log_path), job_id)
        )

    def get_job(self, job_id: int) -> Job | None:
        row = self.connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self,
        limit: int | None = None,
        *,
        group: str | None = None,
        states: Sequence[JobState] | None = None,
        id_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> list[Job]:
        """Newest first. Filters combine with AND; ranges are inclusive and combine with OR."""
        clauses: list[str] = []
        params: list[object] = []
        if group is not None:
            clauses.append("group_name = ?")
            params.append(group)
        if states:
            clauses.append(f"state IN ({','.join('?' for _ in states)})")
            params.extend(state.value for state in states)
        if id_ranges:
            clauses.append("(" + " OR ".join("id BETWEEN ? AND ?" for _ in id_ranges) + ")")
            for low, high in id_ranges:
                params.extend((low, high))
        sql = "SELECT * FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_job(row) for row in self.connection.execute(sql, params)]

    def waiting_jobs(self) -> list[Job]:
        rows = self.connection.execute(
            "SELECT * FROM jobs WHERE state = ? ORDER BY submit_time, id",
            (JobState.WAITING.value,),
        )
        return [self._row_to_job(row) for row in rows]

    def active_jobs(self) -> list[Job]:
        states = (JobState.STARTING.value, JobState.RUNNING.value, JobState.CANCELLING.value)
        placeholders = ",".join("?" for _ in states)
        rows = self.connection.execute(
            f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY id", states
        )
        return [self._row_to_job(row) for row in rows]

    def reserve(self, job_id: int, gpus: Sequence[GPUObservation]) -> None:
        """Atomically move WAITING -> STARTING and record the complete assignment."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET state = ?, start_time = ? WHERE id = ? AND state = ?",
                (JobState.STARTING.value, utc_now(), job_id, JobState.WAITING.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"job {job_id} is no longer waiting")
            conn.executemany(
                "INSERT INTO job_gpus(job_id, gpu_uuid, gpu_index_at_start) VALUES (?, ?, ?)",
                [(job_id, gpu.uuid, gpu.index) for gpu in gpus],
            )

    def mark_running(
        self, job_id: int, pid: int, pgid: int, start_time: int | None, boot: str | None
    ) -> None:
        with self.transaction() as conn:
            cursor = conn.execute(
                """UPDATE jobs SET state = ?, pid = ?, pgid = ?, process_start_time = ?,
                   boot_id = ? WHERE id = ? AND state = ?""",
                (
                    JobState.RUNNING.value,
                    pid,
                    pgid,
                    start_time,
                    boot,
                    job_id,
                    JobState.STARTING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"job {job_id} is not starting")

    def set_cancelling(self, job_id: int) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET state = ? WHERE id = ? AND state IN (?, ?)",
                (
                    JobState.CANCELLING.value,
                    job_id,
                    JobState.STARTING.value,
                    JobState.RUNNING.value,
                ),
            )
            return cursor.rowcount == 1

    def finish(
        self,
        job_id: int,
        state: JobState,
        *,
        exit_code: int | None = None,
        reason: str | None = None,
    ) -> None:
        if not state.terminal:
            raise ValueError(f"{state} is not terminal")
        with self.transaction() as conn:
            conn.execute(
                """UPDATE jobs SET state = ?, end_time = ?, exit_code = ?, failure_reason = ?
                   WHERE id = ?""",
                (state.value, utc_now(), exit_code, reason, job_id),
            )

    def assignments(self) -> dict[str, int]:
        rows = self.connection.execute(
            """SELECT g.gpu_uuid, g.job_id FROM job_gpus g JOIN jobs j ON j.id = g.job_id
               WHERE j.state IN (?, ?, ?, ?)""",
            (
                JobState.STARTING.value,
                JobState.RUNNING.value,
                JobState.CANCELLING.value,
                JobState.ORPHANED.value,
            ),
        )
        return {str(row["gpu_uuid"]): int(row["job_id"]) for row in rows}

    def terminal_jobs_before(
        self, cutoff: str, states: Sequence[JobState] | None = None
    ) -> list[Job]:
        """Terminal jobs that ended before `cutoff`.

        Timestamps are all produced by `utc_now()` in one fixed ISO-8601 UTC format,
        so a lexicographic comparison is also a chronological one.
        """
        wanted = [state for state in (states or list(JobState)) if state.terminal]
        if not wanted:
            return []
        placeholders = ",".join("?" for _ in wanted)
        rows = self.connection.execute(
            f"""SELECT * FROM jobs WHERE state IN ({placeholders})
                AND end_time IS NOT NULL AND end_time < ? ORDER BY id""",
            (*(state.value for state in wanted), cutoff),
        )
        return [self._row_to_job(row) for row in rows]

    def delete_jobs(self, job_ids: Sequence[int]) -> list[Job]:
        """Delete terminal jobs, returning the removed rows so callers can drop logs.

        Refuses the whole batch if any job is missing or still active, so a typo in a
        job list cannot partially destroy history. `job_gpus` follows via ON DELETE
        CASCADE.
        """
        removed: list[Job] = []
        with self.transaction() as conn:
            for job_id in job_ids:
                job = self.get_job(job_id)
                if job is None:
                    raise KeyError(f"job {job_id} does not exist")
                if not job.state.terminal:
                    raise ValueError(
                        f"job {job_id} is {job.state.value} and cannot be deleted; cancel it first"
                    )
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                removed.append(job)
        return removed

    def newer_attempt(self, root: int, after: int) -> int | None:
        """The id of an attempt in `root`'s retry lineage submitted after job `after`."""
        row = self.connection.execute(
            "SELECT id FROM jobs WHERE (id = ? OR retry_of = ?) AND id > ? ORDER BY id LIMIT 1",
            (root, root, after),
        ).fetchone()
        return int(row["id"]) if row else None

    # A job is superseded once a later attempt in its retry lineage exists.
    _SUPERSEDED = """EXISTS (SELECT 1 FROM jobs later
                             WHERE later.retry_of = COALESCE(jobs.retry_of, jobs.id)
                             AND later.id > jobs.id)"""

    def group_counts(self) -> dict[str, dict[str, int]]:
        """Per-group state counts over each lineage's latest attempt only.

        A cell that failed once and then succeeded on retry counts as DONE, not as one
        DONE and one FAILED, which would misreport how much of a sweep is broken.
        """
        counts: dict[str, dict[str, int]] = {}
        rows = self.connection.execute(
            f"""SELECT group_name, state, COUNT(*) AS n FROM jobs
                WHERE group_name IS NOT NULL AND NOT {self._SUPERSEDED}
                GROUP BY group_name, state"""
        )
        for row in rows:
            counts.setdefault(str(row["group_name"]), {})[str(row["state"])] = int(row["n"])
        return counts

    def group_retried_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            f"""SELECT group_name, COUNT(*) AS n FROM jobs
                WHERE group_name IS NOT NULL AND {self._SUPERSEDED} GROUP BY group_name"""
        )
        return {str(row["group_name"]): int(row["n"]) for row in rows}

    def held_groups(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT group_name, reason FROM group_holds WHERE held = 1")
        return {str(row["group_name"]): str(row["reason"] or "") for row in rows}

    def set_group_hold(self, group: str, held: bool, reason: str | None = None) -> bool:
        """Hold or release a group. Returns False if it was already in that state."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT held FROM group_holds WHERE group_name = ?", (group,)
            ).fetchone()
            if row is not None and bool(row["held"]) == held:
                return False
            if row is None and not held:
                return False
            conn.execute(
                """INSERT INTO group_holds (group_name, held, reason, changed_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(group_name) DO UPDATE SET
                     held = excluded.held, reason = excluded.reason,
                     changed_at = excluded.changed_at""",
                (group, int(held), reason, utc_now()),
            )
        return True

    def group_released_at(self, group: str) -> str | None:
        row = self.connection.execute(
            "SELECT changed_at FROM group_holds WHERE group_name = ? AND held = 0", (group,)
        ).fetchone()
        return str(row["changed_at"]) if row else None

    def recent_group_outcomes(self, group: str, since: str | None, limit: int) -> list[Job]:
        """The group's most recent DONE or FAILED jobs, newest first.

        Cancelled jobs are skipped: cancelling says nothing about whether the batch works.
        """
        rows = self.connection.execute(
            """SELECT * FROM jobs WHERE group_name = ? AND state IN (?, ?)
               AND end_time IS NOT NULL AND (? IS NULL OR end_time > ?)
               ORDER BY end_time DESC, id DESC LIMIT ?""",
            (group, JobState.DONE.value, JobState.FAILED.value, since, since, limit),
        )
        return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        gpu_rows = self.connection.execute(
            "SELECT gpu_uuid, gpu_index_at_start FROM job_gpus WHERE job_id = ?"
            " ORDER BY gpu_index_at_start",
            (row["id"],),
        ).fetchall()
        return Job(
            id=int(row["id"]),
            name=row["name"],
            state=JobState(row["state"]),
            argv=list(json.loads(row["command_json"])),
            cwd=Path(row["cwd"]),
            env=dict(json.loads(row["env_json"])),
            requested_gpus=int(row["requested_gpus"]),
            submit_time=row["submit_time"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            pid=row["pid"],
            pgid=row["pgid"],
            process_start_time=row["process_start_time"],
            boot_id=row["boot_id"],
            exit_code=row["exit_code"],
            log_path=Path(row["log_path"]) if row["log_path"] else None,
            failure_reason=row["failure_reason"],
            gpu_uuids=[str(g["gpu_uuid"]) for g in gpu_rows],
            gpu_indices=[int(g["gpu_index_at_start"]) for g in gpu_rows],
            group=row["group_name"],
            key=row["job_key"],
            retry_of=row["retry_of"],
        )
