from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricCollector(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...
