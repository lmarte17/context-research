from __future__ import annotations

from dataclasses import asdict
from typing import Any

from context_research.serving.backends.base import GenerationRequest
from context_research.serving.broker.pd_broker import LocalQueuePDBroker
from context_research.serving.scheduler.base import RouteEvent, Scheduler


def _prompt_token_estimate(prompt: str) -> int:
    stripped = prompt.strip()
    if not stripped:
        return 0
    return len(stripped.split())


class DisaggregatedScheduler(Scheduler):
    def __init__(
        self,
        *,
        prefill_pool: str = "prefill",
        prefill_pool_size: int = 1,
        decode_pool: str = "decode",
        decode_pool_size: int = 1,
        broker: LocalQueuePDBroker | None = None,
    ) -> None:
        self._prefill_pool = prefill_pool
        self._prefill_pool_size = max(1, prefill_pool_size)
        self._decode_pool = decode_pool
        self._decode_pool_size = max(1, decode_pool_size)
        self._broker = broker or LocalQueuePDBroker()
        self._route_log: list[RouteEvent] = []

    @property
    def broker(self) -> LocalQueuePDBroker:
        return self._broker

    def route_prefill(self, request: GenerationRequest) -> str:
        if request.request_id is None:
            raise ValueError(
                "DisaggregatedScheduler requires request.request_id for handoff routing."
            )
        request_id = request.request_id
        prompt_tokens = _prompt_token_estimate(request.prompt)

        handoff = self._broker.register_prefill_completion(
            request_id=request_id,
            prefill_pool=self._prefill_pool,
            decode_pool=self._decode_pool,
            prompt_tokens=prompt_tokens,
            metadata={"max_new_tokens": request.max_new_tokens},
        )

        self._route_log.append(
            RouteEvent.create(
                request_id=request_id,
                phase="prefill",
                pool=self._prefill_pool,
                metadata={
                    "handoff_id": handoff.handoff_id,
                    "prompt_tokens": prompt_tokens,
                    "transfer_bytes": handoff.transfer_bytes,
                    "pool_size": self._prefill_pool_size,
                },
            )
        )
        return self._prefill_pool

    def route_decode(self, request_id: str) -> str:
        handoff = self._broker.claim_decode_handoff(request_id)
        metadata: dict[str, Any]
        if handoff is None:
            metadata = {"handoff_missing": True, "pool_size": self._decode_pool_size}
        else:
            metadata = {
                "handoff_id": handoff.handoff_id,
                "transfer_bytes": handoff.transfer_bytes,
                "transfer_ms": handoff.transfer_ms,
                "pool_size": self._decode_pool_size,
            }

        self._route_log.append(
            RouteEvent.create(
                request_id=request_id,
                phase="decode",
                pool=self._decode_pool,
                metadata=metadata,
            )
        )
        return self._decode_pool

    def route_log(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self._route_log]

    def clear_route_log(self) -> None:
        self._route_log.clear()
