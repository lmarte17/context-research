from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class HandoffRecord:
    handoff_id: str
    request_id: str
    prefill_pool: str
    decode_pool: str
    prompt_tokens: int
    transfer_bytes: int
    enqueued_at_utc: str
    dequeued_at_utc: str | None = None
    transfer_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _enqueued_perf: float = field(default=0.0, repr=False)


class LocalQueuePDBroker:
    def __init__(self, bytes_per_token: int = 2048) -> None:
        self._bytes_per_token = bytes_per_token
        self._prefill_queue: deque[str] = deque()
        self._decode_queue: deque[str] = deque()
        self._pending: dict[str, HandoffRecord] = {}
        self._handoff_by_request: dict[str, str] = {}
        self._completed: list[HandoffRecord] = []
        self._total_transfer_bytes = 0

    def register_prefill_completion(
        self,
        *,
        request_id: str,
        prefill_pool: str,
        decode_pool: str,
        prompt_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> HandoffRecord:
        handoff_id = f"handoff-{uuid.uuid4().hex[:12]}"
        transfer_bytes = max(prompt_tokens, 0) * self._bytes_per_token
        record = HandoffRecord(
            handoff_id=handoff_id,
            request_id=request_id,
            prefill_pool=prefill_pool,
            decode_pool=decode_pool,
            prompt_tokens=prompt_tokens,
            transfer_bytes=transfer_bytes,
            enqueued_at_utc=_utc_now(),
            metadata=metadata or {},
            _enqueued_perf=perf_counter(),
        )

        self._prefill_queue.append(handoff_id)
        self._decode_queue.append(handoff_id)
        self._pending[handoff_id] = record
        self._handoff_by_request[request_id] = handoff_id
        self._total_transfer_bytes += transfer_bytes
        return record

    def claim_decode_handoff(self, request_id: str) -> HandoffRecord | None:
        handoff_id = self._handoff_by_request.pop(request_id, None)
        if handoff_id is None:
            return None

        record = self._pending.pop(handoff_id, None)
        if record is None:
            return None

        try:
            self._decode_queue.remove(handoff_id)
        except ValueError:
            pass
        try:
            self._prefill_queue.remove(handoff_id)
        except ValueError:
            pass

        record.dequeued_at_utc = _utc_now()
        record.transfer_ms = (perf_counter() - record._enqueued_perf) * 1000.0
        self._completed.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        total_transfer_ms = sum(
            record.transfer_ms or 0.0 for record in self._completed
        )
        completed_count = len(self._completed)
        avg_transfer_ms = total_transfer_ms / completed_count if completed_count else 0.0

        return {
            "prefill_queue_depth": len(self._prefill_queue),
            "decode_queue_depth": len(self._decode_queue),
            "pending_handoffs": len(self._pending),
            "completed_handoffs": completed_count,
            "total_transfer_bytes": self._total_transfer_bytes,
            "total_transfer_ms": total_transfer_ms,
            "avg_transfer_ms": avg_transfer_ms,
        }

    def completed_handoffs(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for record in self._completed:
            serialized = asdict(record)
            serialized.pop("_enqueued_perf", None)
            payload.append(serialized)
        return payload
