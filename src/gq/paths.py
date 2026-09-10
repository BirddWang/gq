from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    runtime_dir: Path
    database: Path
    logs_dir: Path
    daemon_log: Path
    socket: Path
    pid_file: Path
    lock_file: Path

    @classmethod
    def from_environment(cls) -> Paths:
        data_home = Path(
            os.environ.get("GQ_DATA_DIR")
            or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "gq"
        ).expanduser()
        runtime_override = os.environ.get("GQ_RUNTIME_DIR")
        if runtime_override:
            runtime_dir = Path(runtime_override).expanduser()
        elif os.environ.get("XDG_RUNTIME_DIR"):
            runtime_dir = Path(os.environ["XDG_RUNTIME_DIR"]) / "gq"
        else:
            runtime_dir = Path.home() / ".local/run/gq"
        database = Path(os.environ.get("GQ_DATABASE", data_home / "gq.sqlite3")).expanduser()
        return cls(
            data_dir=data_home,
            runtime_dir=runtime_dir,
            database=database,
            logs_dir=data_home / "logs",
            daemon_log=data_home / "gq-daemon.log",
            socket=runtime_dir / "gq.sock",
            pid_file=runtime_dir / "gq.pid",
            lock_file=runtime_dir / "gq.lock",
        )

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.logs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.data_dir, 0o700)
        os.chmod(self.logs_dir, 0o700)
        os.chmod(self.runtime_dir, 0o700)
