from __future__ import annotations

import contextlib
from typing import Protocol

from .models import ComputeProcess, GPUObservation


class NVMLUnavailable(RuntimeError):
    pass


class GPUProvider(Protocol):
    def snapshot(self) -> list[GPUObservation]: ...

    def close(self) -> None: ...


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


class NVMLProvider:
    def __init__(self) -> None:
        try:
            import pynvml
        except ImportError as exc:  # pragma: no cover - dependency is installed normally
            raise NVMLUnavailable("nvidia-ml-py is not installed") from exc
        self.nvml = pynvml
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            raise NVMLUnavailable(f"NVML initialization failed: {exc}") from exc
        self._closed = False

    def snapshot(self) -> list[GPUObservation]:
        n = self.nvml
        try:
            count = n.nvmlDeviceGetCount()
        except n.NVMLError as exc:
            raise NVMLUnavailable(f"NVML device discovery failed: {exc}") from exc
        observations: list[GPUObservation] = []
        for index in range(count):
            try:
                handle = n.nvmlDeviceGetHandleByIndex(index)
                uuid = _decode(n.nvmlDeviceGetUUID(handle))
                name = _decode(n.nvmlDeviceGetName(handle))
                memory = n.nvmlDeviceGetMemoryInfo(handle)
                raw_processes = n.nvmlDeviceGetComputeRunningProcesses(handle)
                processes: list[ComputeProcess] = []
                for proc in raw_processes:
                    proc_name: str | None = None
                    with contextlib.suppress(n.NVMLError):
                        proc_name = _decode(n.nvmlSystemGetProcessName(proc.pid))
                    used = getattr(proc, "usedGpuMemory", None)
                    # NVML uses an unsigned sentinel when memory is unavailable.
                    if used is not None and used > (1 << 60):
                        used = None
                    processes.append(ComputeProcess(proc.pid, used, proc_name))
                observations.append(
                    GPUObservation(
                        uuid=uuid,
                        index=index,
                        name=name,
                        total_memory=memory.total,
                        used_memory=memory.used,
                        compute_processes=tuple(processes),
                    )
                )
            except n.NVMLError as exc:
                # Retain a stable per-index placeholder and fail closed.
                observations.append(
                    GPUObservation(
                        uuid=f"UNKNOWN-INDEX-{index}",
                        index=index,
                        name="unknown",
                        error=str(exc),
                    )
                )
        return observations

    def close(self) -> None:
        if not self._closed:
            try:
                self.nvml.nvmlShutdown()
            finally:
                self._closed = True
