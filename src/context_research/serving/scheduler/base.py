from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from context_research.serving.backends.base import GenerationRequest


@dataclass(frozen=True)
class RouteEvent:
    request_id: str
    phase: str
    pool: str
    created_at_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        request_id: str,
        phase: str,
        pool: str,
        metadata: dict[str, Any] | None = None,
    ) -> "RouteEvent":
        return cls(
            request_id=request_id,
            phase=phase,
            pool=pool,
            created_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            metadata=metadata or {},
        )


@runtime_checkable
class Scheduler(Protocol):
    def route_prefill(self, request: GenerationRequest) -> str:
        ...

    def route_decode(self, request_id: str) -> str:
        ...

    def route_log(self) -> list[dict[str, Any]]:
        ...
