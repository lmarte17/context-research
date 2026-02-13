from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable

from context_research.benchmarks.base import BenchmarkSample, LoadedBenchmarkTask
from context_research.benchmarks.infinitybench import default_samples as infinitybench_default_samples
from context_research.benchmarks.longbench import default_samples as longbench_default_samples
from context_research.benchmarks.pg19 import default_samples as pg19_default_samples
from context_research.benchmarks.ruler_subset import default_samples as ruler_default_samples
from context_research.config.io import load_simple_yaml


TaskLoader = Callable[[], list[BenchmarkSample]]

_TASK_REGISTRY: dict[str, TaskLoader] = {
    "pg19_subset": pg19_default_samples,
    "longbench_subset": longbench_default_samples,
    "ruler_subset": ruler_default_samples,
    "infinitybench_subset": infinitybench_default_samples,
}


@dataclass(frozen=True)
class LoadedBenchmarkSuite:
    name: str
    config_path: str
    seed: int
    sample_count: int
    tasks: list[LoadedBenchmarkTask]


def load_benchmark_suite(config_path: str | Path) -> LoadedBenchmarkSuite:
    payload = load_simple_yaml(config_path)
    suite_name = _safe_str(payload.get("name"), "benchmark_suite")
    seed = _safe_int(payload.get("seed"), 42)
    sample_count = max(1, _safe_int(payload.get("sample_count"), 8))

    task_entries = payload.get("tasks")
    if not isinstance(task_entries, list) or not task_entries:
        raise ValueError("Benchmark config must define a non-empty 'tasks' list.")

    dataset_paths = _safe_mapping(payload.get("dataset_paths"))
    task_sample_counts = _safe_mapping(payload.get("task_sample_counts"))
    task_seeds = _safe_mapping(payload.get("task_seeds"))
    tasks: list[LoadedBenchmarkTask] = []
    for raw_entry in task_entries:
        task_name, task_options = _parse_task_entry(raw_entry)
        dataset_path = _safe_str(
            task_options.get("dataset_path", dataset_paths.get(task_name)),
            "",
        )
        task_sample_count = max(
            1,
            _safe_int(
                task_options.get("sample_count", task_sample_counts.get(task_name)),
                sample_count,
            ),
        )
        task_seed = _safe_int(task_options.get("seed", task_seeds.get(task_name)), seed)
        all_samples, source = _load_task_samples(
            task_name=task_name,
            dataset_path=dataset_path or None,
        )
        subset = deterministic_subset(
            samples=all_samples,
            sample_count=task_sample_count,
            seed=task_seed,
        )
        tasks.append(
            LoadedBenchmarkTask(
                name=task_name,
                samples=subset,
                source=source,
            )
        )

    return LoadedBenchmarkSuite(
        name=suite_name,
        config_path=str(Path(config_path).resolve()),
        seed=seed,
        sample_count=sample_count,
        tasks=tasks,
    )


def deterministic_subset(
    *,
    samples: list[BenchmarkSample],
    sample_count: int,
    seed: int,
) -> list[BenchmarkSample]:
    if sample_count <= 0:
        return []
    if len(samples) <= sample_count:
        return list(samples)

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected_indices = indices[:sample_count]
    return [samples[index] for index in selected_indices]


def _load_task_samples(*, task_name: str, dataset_path: str | None) -> tuple[list[BenchmarkSample], str]:
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset path does not exist: {path}")
        samples = _load_jsonl_samples(task_name=task_name, dataset_path=path)
        return (samples, "jsonl")

    loader = _TASK_REGISTRY.get(task_name)
    if loader is None:
        known = ", ".join(sorted(_TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown benchmark task '{task_name}'. Known tasks: {known}")
    return (loader(), "builtin")


def _load_jsonl_samples(*, task_name: str, dataset_path: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in benchmark dataset {dataset_path}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_no} in benchmark dataset {dataset_path}."
                )

            prompt = _safe_str(payload.get("prompt"), "")
            reference = _safe_str(
                payload.get("reference", payload.get("answer", payload.get("target"))),
                "",
            )
            sample_id = _safe_str(payload.get("sample_id"), f"{task_name}-{line_no:05d}")
            if not prompt or not reference:
                raise ValueError(
                    f"Dataset row {line_no} in {dataset_path} must contain prompt/reference fields."
                )
            metadata_raw = payload.get("metadata")
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            samples.append(
                BenchmarkSample(
                    task=task_name,
                    sample_id=sample_id,
                    prompt=prompt,
                    reference=reference,
                    metadata=metadata,
                )
            )

    if not samples:
        raise ValueError(f"No benchmark samples found in dataset file {dataset_path}.")
    return samples


def _parse_task_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        name = entry.strip()
        if not name:
            raise ValueError("Benchmark task names must be non-empty strings.")
        return (name, {})

    if isinstance(entry, dict):
        name = _safe_str(entry.get("name"), "")
        if not name:
            raise ValueError("Benchmark task mappings must include non-empty 'name'.")
        options = {key: value for key, value in entry.items() if key != "name"}
        return (name, options)

    raise ValueError("Benchmark tasks must be either string names or mappings.")


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _safe_int(value: Any, default: int) -> int:
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
    return default

