from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ServingBackend(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def warmup(self) -> None:
        ...

    def prefill(self, request: GenerationRequest) -> dict[str, Any]:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
