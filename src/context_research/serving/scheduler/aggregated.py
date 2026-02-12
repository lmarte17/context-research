from __future__ import annotations

from dataclasses import asdict
from typing import Any
import uuid

from context_research.serving.backends.base import GenerationRequest
from context_research.serving.scheduler.base import RouteEvent, Scheduler


class AggregatedScheduler(Scheduler):
    def __init__(self, pool_name: str = "aggregated", pool_size: int = 1) -> None:
        self._pool_name = pool_name
        self._pool_size = max(1, pool_size)
        self._route_log: list[RouteEvent] = []

    def route_prefill(self, request: GenerationRequest) -> str:
        request_id = request.request_id or self._new_request_id()
        self._route_log.append(
            RouteEvent.create(
                request_id=request_id,
                phase="prefill",
                pool=self._pool_name,
                metadata={
                    "prompt_chars": len(request.prompt),
                    "max_new_tokens": request.max_new_tokens,
                    "pool_size": self._pool_size,
                },
            )
        )
        return self._pool_name

    def route_decode(self, request_id: str) -> str:
        self._route_log.append(
            RouteEvent.create(
                request_id=request_id,
                phase="decode",
                pool=self._pool_name,
                metadata={"pool_size": self._pool_size},
            )
        )
        return self._pool_name

    def route_log(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self._route_log]

    def clear_route_log(self) -> None:
        self._route_log.clear()

    def _new_request_id(self) -> str:
        return f"req-{uuid.uuid4().hex[:10]}"
