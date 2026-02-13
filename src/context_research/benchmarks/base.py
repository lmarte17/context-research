from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSample:
    task: str
    sample_id: str
    prompt: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedBenchmarkTask:
    name: str
    samples: list[BenchmarkSample]
    source: str

