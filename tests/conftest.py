from __future__ import annotations

from dataclasses import dataclass

import pytest

from gq.models import GPUObservation


@dataclass
class FakeGPUProvider:
    observations: list[GPUObservation]
    error: Exception | None = None
    closed: bool = False

    def snapshot(self) -> list[GPUObservation]:
        if self.error:
            raise self.error
        return list(self.observations)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def gpu_factory():
    def make(count: int) -> list[GPUObservation]:
        return [
            GPUObservation(uuid=f"GPU-{index}", index=index, name="Fake GPU")
            for index in range(count)
        ]

    return make
