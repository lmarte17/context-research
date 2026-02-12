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


@dataclass(frozen=True)
class KVTransferSample:
    request_id: str
    handoff_id: str
    transfer_bytes: int
    transfer_ms: float
    prefill_pool: str | None = None
    decode_pool: str | None = None


class KVTransferStatsCollector:
    def __init__(self) -> None:
        self._samples: list[KVTransferSample] = []
        self._started = False
        self._started_perf: float | None = None
        self._stopped_perf: float | None = None
        self._mode = "aggregated"
        self._wall_time_ms: float | None = None

    def start(self) -> None:
        self._samples = []
        self._started = True
        self._started_perf = perf_counter()
        self._stopped_perf = None
        self._wall_time_ms = None

    def stop(self) -> None:
        if not self._started or self._stopped_perf is not None:
            return
        self._stopped_perf = perf_counter()

    def set_mode(self, mode: str) -> None:
        self._mode = mode.strip().lower()

    def set_wall_time_ms(self, wall_time_ms: float) -> None:
        self._wall_time_ms = _safe_float(wall_time_ms)

    def ingest_completed_handoffs(self, handoffs: list[dict[str, Any]]) -> None:
        for handoff in handoffs:
            transfer_ms = _safe_float(handoff.get("transfer_ms"))
            transfer_bytes = _safe_int(handoff.get("transfer_bytes"))
            if transfer_ms is None or transfer_bytes is None:
                continue

            request_id = str(handoff.get("request_id", "")).strip()
            handoff_id = str(handoff.get("handoff_id", "")).strip()
            if not request_id or not handoff_id:
                continue

            self._samples.append(
                KVTransferSample(
                    request_id=request_id,
                    handoff_id=handoff_id,
                    transfer_bytes=max(transfer_bytes, 0),
                    transfer_ms=max(transfer_ms, 0.0),
                    prefill_pool=_safe_optional_str(handoff.get("prefill_pool")),
                    decode_pool=_safe_optional_str(handoff.get("decode_pool")),
                )
            )

    def snapshot(self) -> dict[str, Any]:
        total_transfer_bytes = sum(sample.transfer_bytes for sample in self._samples)
        total_transfer_ms = sum(sample.transfer_ms for sample in self._samples)
        throughput_mb_per_s: float | None = None
        if total_transfer_ms > 0.0:
            throughput_mb_per_s = (total_transfer_bytes / (1024.0 * 1024.0)) / (
                total_transfer_ms / 1000.0
            )

        available = self._mode == "disaggregated"
        stall_ratio: float | None = None
        if available and self._wall_time_ms is not None and self._wall_time_ms > 0.0:
            stall_ratio = total_transfer_ms / self._wall_time_ms

        window_ms: float | None = None
        if self._started_perf is not None:
            end = self._stopped_perf if self._stopped_perf is not None else perf_counter()
            window_ms = max((end - self._started_perf) * 1000.0, 0.0)

        return {
            "collector": "kv_transfer_stats",
            "started": self._started,
            "mode": self._mode,
            "available": available,
            "availability_reason": None if available else "not_disaggregated_mode",
            "window_ms": window_ms,
            "sample_count": len(self._samples),
            "samples": [asdict(sample) for sample in self._samples],
            "summary": {
                "total_transfer_bytes": total_transfer_bytes,
                "total_transfer_ms": total_transfer_ms,
                "avg_transfer_ms": (
                    total_transfer_ms / len(self._samples) if self._samples else None
                ),
                "transfer_throughput_mb_per_s": throughput_mb_per_s,
                "stall_ratio": stall_ratio,
                "stall_ratio_basis": (
                    "total_transfer_ms / run_wall_time_ms"
                    if available and stall_ratio is not None
                    else None
                ),
            },
        }


def _safe_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None
