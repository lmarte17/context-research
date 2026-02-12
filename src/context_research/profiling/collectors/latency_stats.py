from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit())):
            return int(stripped)
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (q / 100.0)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = position - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return lower + (upper - lower) * weight


@dataclass(frozen=True)
class LatencySample:
    request_id: str
    ttft_ms: float
    tpot_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] | None = None


class LatencyStatsCollector:
    def __init__(self) -> None:
        self._samples: list[LatencySample] = []
        self._started = False
        self._started_perf: float | None = None
        self._stopped_perf: float | None = None

    def start(self) -> None:
        self._samples = []
        self._started = True
        self._started_perf = perf_counter()
        self._stopped_perf = None

    def stop(self) -> None:
        if not self._started or self._stopped_perf is not None:
            return
        self._stopped_perf = perf_counter()

    def record_sample(
        self,
        *,
        request_id: str,
        ttft_ms: float,
        tpot_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ttft_value = _safe_float(ttft_ms)
        tpot_value = _safe_float(tpot_ms)
        if ttft_value is None or tpot_value is None:
            return

        self._samples.append(
            LatencySample(
                request_id=request_id,
                ttft_ms=ttft_value,
                tpot_ms=tpot_value,
                prompt_tokens=_safe_int(prompt_tokens),
                completion_tokens=_safe_int(completion_tokens),
                metadata=metadata or {},
            )
        )

    def snapshot(self) -> dict[str, Any]:
        ttft_values = [sample.ttft_ms for sample in self._samples]
        tpot_values = [sample.tpot_ms for sample in self._samples]

        window_ms: float | None = None
        if self._started_perf is not None:
            end = self._stopped_perf if self._stopped_perf is not None else perf_counter()
            window_ms = max((end - self._started_perf) * 1000.0, 0.0)

        return {
            "collector": "latency_stats",
            "started": self._started,
            "window_ms": window_ms,
            "sample_count": len(self._samples),
            "samples": [asdict(sample) for sample in self._samples],
            "summary": {
                "ttft_ms": self._distribution(ttft_values),
                "tpot_ms": self._distribution(tpot_values),
            },
        }

    def _distribution(self, values: list[float]) -> dict[str, float | None]:
        if not values:
            return {
                "min": None,
                "max": None,
                "avg": None,
                "p50": None,
                "p95": None,
            }

        avg = sum(values) / len(values)
        return {
            "min": min(values),
            "max": max(values),
            "avg": avg,
            "p50": _percentile(values, 50.0),
            "p95": _percentile(values, 95.0),
        }
