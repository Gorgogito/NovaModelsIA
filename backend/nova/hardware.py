"""Detección de hardware: CPU, RAM y GPU NVIDIA (si existe)."""
from __future__ import annotations

import psutil
from pydantic import BaseModel


class GpuInfo(BaseModel):
    name: str
    vram_total_mb: int
    vram_used_mb: int
    utilization_pct: int


class HardwareStatus(BaseModel):
    cpu_pct: float
    cpu_cores: int
    ram_total_mb: int
    ram_used_mb: int
    gpus: list[GpuInfo]


def _read_gpus() -> list[GpuInfo]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return []
    gpus: list[GpuInfo] = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append(
                GpuInfo(
                    name=name.decode() if isinstance(name, bytes) else name,
                    vram_total_mb=mem.total // 2**20,
                    vram_used_mb=mem.used // 2**20,
                    utilization_pct=util.gpu,
                )
            )
    finally:
        pynvml.nvmlShutdown()
    return gpus


def read_status() -> HardwareStatus:
    vm = psutil.virtual_memory()
    return HardwareStatus(
        cpu_pct=psutil.cpu_percent(interval=None),
        cpu_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_mb=vm.total // 2**20,
        ram_used_mb=vm.used // 2**20,
        gpus=_read_gpus(),
    )
