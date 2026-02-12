from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from time import perf_counter, sleep
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class GPUSample:
    timestamp_utc: str
    elapsed_ms: float
    index: int
    memory_used_mb: float
    memory_total_mb: float
    gpu_utilization_pct: float
    memory_utilization_pct: float


@dataclass(frozen=True)
class GPUDeviceInfo:
    index: int
    name: str
    memory_total_mb: float


class GPUStatsCollector:
    def __init__(self, sample_interval_ms: int = 100) -> None:
        self._sample_interval_s = max(sample_interval_ms, 20) / 1000.0
        self._samples: list[GPUSample] = []
        self._devices: dict[int, GPUDeviceInfo] = {}
        self._available = False
        self._availability_reason: str | None = None
        self._started = False
        self._started_perf: float | None = None
        self._stopped_perf: float | None = None
        self._worker: Thread | None = None
        self._stop_event = Event()
        self._lock = Lock()
        self._nvml_handles: list[Any] = []
        self._nvml: dict[str, Any] = {}

    def start(self) -> None:
        self._samples = []
        self._devices = {}
        self._available = False
        self._availability_reason = None
        self._started = True
        self._started_perf = perf_counter()
        self._stopped_perf = None
        self._stop_event.clear()

        self._initialize_nvml()
        if not self._available:
            return

        self._collect_once()
        self._worker = Thread(target=self._sample_loop, name="gpu-stats-collector", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if not self._started or self._stopped_perf is not None:
            return

        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

        self._collect_once()
        self._shutdown_nvml()
        self._stopped_perf = perf_counter()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)

        per_gpu_samples: dict[int, list[GPUSample]] = {}
        for sample in samples:
            per_gpu_samples.setdefault(sample.index, []).append(sample)

        per_gpu: dict[str, Any] = {}
        for gpu_index, gpu_samples in per_gpu_samples.items():
            memory_used = [sample.memory_used_mb for sample in gpu_samples]
            gpu_util = [sample.gpu_utilization_pct for sample in gpu_samples]
            memory_util = [sample.memory_utilization_pct for sample in gpu_samples]
            device_info = self._devices.get(gpu_index)

            per_gpu[str(gpu_index)] = {
                "index": gpu_index,
                "name": device_info.name if device_info is not None else None,
                "memory_total_mb": (
                    device_info.memory_total_mb if device_info is not None else None
                ),
                "sample_count": len(gpu_samples),
                "memory_used_mb": {
                    "avg": _avg(memory_used),
                    "peak": max(memory_used) if memory_used else None,
                },
                "gpu_utilization_pct": {
                    "avg": _avg(gpu_util),
                    "peak": max(gpu_util) if gpu_util else None,
                },
                "memory_utilization_pct": {
                    "avg": _avg(memory_util),
                    "peak": max(memory_util) if memory_util else None,
                },
            }

        run_summary = {
            "memory_used_mb_avg": _avg([sample.memory_used_mb for sample in samples]),
            "memory_used_mb_peak": (
                max(sample.memory_used_mb for sample in samples) if samples else None
            ),
            "gpu_utilization_pct_avg": _avg([sample.gpu_utilization_pct for sample in samples]),
            "gpu_utilization_pct_peak": (
                max(sample.gpu_utilization_pct for sample in samples) if samples else None
            ),
            "memory_utilization_pct_avg": _avg(
                [sample.memory_utilization_pct for sample in samples]
            ),
            "memory_utilization_pct_peak": (
                max(sample.memory_utilization_pct for sample in samples) if samples else None
            ),
        }

        window_ms: float | None = None
        if self._started_perf is not None:
            end = self._stopped_perf if self._stopped_perf is not None else perf_counter()
            window_ms = max((end - self._started_perf) * 1000.0, 0.0)

        return {
            "collector": "gpu_stats",
            "started": self._started,
            "available": self._available,
            "availability_reason": self._availability_reason,
            "window_ms": window_ms,
            "sample_interval_ms": self._sample_interval_s * 1000.0,
            "sample_count": len(samples),
            "devices": [asdict(device) for _, device in sorted(self._devices.items())],
            "run_summary": run_summary,
            "per_gpu": per_gpu,
            "samples": [asdict(sample) for sample in samples],
        }

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sleep(self._sample_interval_s)
            self._collect_once()

    def _initialize_nvml(self) -> None:
        try:
            from pynvml import (  # type: ignore
                nvmlDeviceGetCount,
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetMemoryInfo,
                nvmlDeviceGetName,
                nvmlInit,
                nvmlShutdown,
            )
        except ImportError:
            self._available = False
            self._availability_reason = "pynvml not installed"
            return

        try:
            nvmlInit()
            device_count = int(nvmlDeviceGetCount())
            handles: list[Any] = []
            devices: dict[int, GPUDeviceInfo] = {}
            for index in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(index)
                memory = nvmlDeviceGetMemoryInfo(handle)
                name_bytes = nvmlDeviceGetName(handle)
                if isinstance(name_bytes, bytes):
                    name = name_bytes.decode("utf-8", errors="replace")
                else:
                    name = str(name_bytes)

                devices[index] = GPUDeviceInfo(
                    index=index,
                    name=name,
                    memory_total_mb=float(memory.total) / (1024.0 * 1024.0),
                )
                handles.append(handle)
        except Exception as exc:
            self._available = False
            self._availability_reason = f"nvml init failed: {exc}"
            try:
                nvmlShutdown()
            except Exception:
                pass
            return

        self._available = True
        self._availability_reason = None
        self._devices = devices
        self._nvml_handles = handles
        self._nvml = {
            "nvmlDeviceGetMemoryInfo": nvmlDeviceGetMemoryInfo,
            "nvmlDeviceGetUtilizationRates": None,
            "nvmlShutdown": nvmlShutdown,
        }

        try:
            from pynvml import nvmlDeviceGetUtilizationRates  # type: ignore

            self._nvml["nvmlDeviceGetUtilizationRates"] = nvmlDeviceGetUtilizationRates
        except ImportError:
            self._nvml["nvmlDeviceGetUtilizationRates"] = None

    def _shutdown_nvml(self) -> None:
        if not self._available:
            return

        shutdown_fn = self._nvml.get("nvmlShutdown")
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception:
                pass
        self._nvml_handles = []

    def _collect_once(self) -> None:
        if not self._available:
            return

        get_memory_info = self._nvml.get("nvmlDeviceGetMemoryInfo")
        get_utilization = self._nvml.get("nvmlDeviceGetUtilizationRates")
        if not callable(get_memory_info):
            return

        started_perf = self._started_perf
        if started_perf is None:
            started_perf = perf_counter()

        new_samples: list[GPUSample] = []
        for index, handle in enumerate(self._nvml_handles):
            try:
                memory_info = get_memory_info(handle)
                memory_used_mb = float(memory_info.used) / (1024.0 * 1024.0)
                memory_total_mb = float(memory_info.total) / (1024.0 * 1024.0)
                gpu_util_pct = 0.0
                memory_util_pct = (memory_used_mb / memory_total_mb * 100.0) if memory_total_mb else 0.0

                if callable(get_utilization):
                    utilization = get_utilization(handle)
                    gpu_util_pct = float(getattr(utilization, "gpu", 0.0))
                    memory_util_pct = float(getattr(utilization, "memory", memory_util_pct))

                new_samples.append(
                    GPUSample(
                        timestamp_utc=_utc_now(),
                        elapsed_ms=max((perf_counter() - started_perf) * 1000.0, 0.0),
                        index=index,
                        memory_used_mb=memory_used_mb,
                        memory_total_mb=memory_total_mb,
                        gpu_utilization_pct=gpu_util_pct,
                        memory_utilization_pct=memory_util_pct,
                    )
                )
            except Exception:
                continue

        if not new_samples:
            return

        with self._lock:
            self._samples.extend(new_samples)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
