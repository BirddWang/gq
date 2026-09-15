from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# The reason recorded when someone holds a group by hand rather than gq holding it
# because its jobs keep failing.
MANUAL_HOLD_REASON = "paused by user"


class JobState(str, Enum):
    WAITING = "WAITING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ORPHANED = "ORPHANED"

    @property
    def terminal(self) -> bool:
        return self in {
            JobState.DONE,
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.ORPHANED,
        }


class GPUState(str, Enum):
    FREE = "FREE"
    RESERVED = "RESERVED"
    RUNNING = "RUNNING"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ComputeProcess:
    pid: int
    used_memory: int | None = None
    name: str | None = None


@dataclass(frozen=True)
class GPUObservation:
    uuid: str
    index: int
    name: str
    total_memory: int | None = None
    used_memory: int | None = None
    compute_processes: tuple[ComputeProcess, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class GPUStatus:
    uuid: str
    index: int
    name: str
    state: GPUState
    owner_job_id: int | None = None
    processes: tuple[ComputeProcess, ...] = ()
    total_memory: int | None = None
    used_memory: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "index": self.index,
            "name": self.name,
            "state": self.state.value,
            "owner_job_id": self.owner_job_id,
            "processes": [
                {"pid": p.pid, "used_memory": p.used_memory, "name": p.name} for p in self.processes
            ],
            "total_memory": self.total_memory,
            "used_memory": self.used_memory,
            "detail": self.detail,
        }


@dataclass
class Job:
    id: int
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    requested_gpus: int
    state: JobState
    submit_time: str
    name: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    pid: int | None = None
    pgid: int | None = None
    process_start_time: int | None = None
    boot_id: str | None = None
    exit_code: int | None = None
    log_path: Path | None = None
    failure_reason: str | None = None
    gpu_uuids: list[str] = field(default_factory=list)
    gpu_indices: list[int] = field(default_factory=list)
    group: str | None = None
    key: str | None = None
    # The first job in this job's retry lineage, so every attempt shares one root.
    retry_of: int | None = None

    def to_dict(self, *, include_env: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "argv": self.argv,
            "cwd": str(self.cwd),
            "requested_gpus": self.requested_gpus,
            "submit_time": self.submit_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "pid": self.pid,
            "pgid": self.pgid,
            "process_start_time": self.process_start_time,
            "boot_id": self.boot_id,
            "exit_code": self.exit_code,
            "log_path": str(self.log_path) if self.log_path else None,
            "failure_reason": self.failure_reason,
            "gpu_uuids": self.gpu_uuids,
            "gpu_indices": self.gpu_indices,
            "group": self.group,
            "key": self.key,
            "retry_of": self.retry_of,
        }
        if include_env:
            result["env"] = self.env
        return result
