from __future__ import annotations

import ctypes
import os
import time

import pytest

from gq.allocator import classify_gpus
from gq.models import GPUState
from gq.nvml import NVMLProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("GQ_RUN_GPU_TESTS") != "1",
    reason="set GQ_RUN_GPU_TESTS=1 to run NVIDIA hardware integration tests",
)


def test_real_nvml_snapshot_has_stable_gpu_identity() -> None:
    provider = NVMLProvider()
    try:
        observations = provider.snapshot()
    finally:
        provider.close()
    assert observations
    assert len({gpu.uuid for gpu in observations}) == len(observations)
    assert [gpu.index for gpu in observations] == list(range(len(observations)))
    assert all(gpu.uuid.startswith("GPU-") for gpu in observations)


def test_idle_external_cuda_context_is_external_until_destroyed() -> None:
    provider = NVMLProvider()
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    context = ctypes.c_void_p()
    try:
        initial = provider.snapshot()
        free = next((gpu for gpu in initial if not gpu.compute_processes), None)
        if free is None:
            pytest.skip("no GPU is free for an external-context integration check")
        assert cuda.cuInit(0) == 0
        device = ctypes.c_int()
        assert cuda.cuDeviceGet(ctypes.byref(device), free.index) == 0
        assert cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0

        deadline = time.monotonic() + 3
        status = None
        while time.monotonic() < deadline:
            observation = next(gpu for gpu in provider.snapshot() if gpu.uuid == free.uuid)
            status = classify_gpus([observation], {}, {}, lambda *_: False)[0]
            if any(proc.pid == os.getpid() for proc in observation.compute_processes):
                break
            time.sleep(0.05)
        assert status is not None and status.state is GPUState.EXTERNAL
        assert any(proc.pid == os.getpid() for proc in status.processes)
    finally:
        if context.value:
            assert cuda.cuCtxDestroy_v2(context) == 0
        provider.close()
