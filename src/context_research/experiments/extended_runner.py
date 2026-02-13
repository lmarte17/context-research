from __future__ import annotations

import math
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from context_research.benchmarks import BenchmarkSample, load_benchmark_suite
from context_research.config.io import load_simple_yaml
from context_research.config.schema import create_run_metadata, run_metadata_to_dict, write_run_metadata
from context_research.experiments import runner as core
from context_research.profiling.collectors import (
    GPUStatsCollector,
    KVTransferStatsCollector,
    LatencyStatsCollector,
)
from context_research.serving import (
    AggregatedScheduler,
    DisaggregatedScheduler,
    GenerationRequest,
    LocalQueuePDBroker,
    VLLMBackend,
    VLLMBackendConfig,
)


@dataclass(frozen=True)
class E3RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


@dataclass(frozen=True)
class E4RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


@dataclass(frozen=True)
class E5RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


@dataclass(frozen=True)
class E6RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


def _to_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _to_plot_value(value: Any) -> float:
    numeric = _to_optional_float(value)
    if numeric is None:
        return float("nan")
    return numeric


def _safe_mode_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return default
    parsed: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip().lower()
            if normalized:
                parsed.append(normalized)
    if not parsed:
        return default
    return parsed


def _safe_average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_delta(lhs: Any, rhs: Any) -> float | None:
    lhs_numeric = _to_optional_float(lhs)
    rhs_numeric = _to_optional_float(rhs)
    if lhs_numeric is None or rhs_numeric is None:
        return None
    return lhs_numeric - rhs_numeric


def _normalize_for_quality(text: str) -> str:
    lowered = text.lower()
    cleaned_chars: list[str] = []
    for char in lowered:
        if char.isalnum() or char in {" ", "-", "_"}:
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
    return " ".join("".join(cleaned_chars).split())


def _token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in _normalize_for_quality(text).split():
        counts[token] = counts.get(token, 0) + 1
    return counts


def _token_f1(prediction: str, reference: str) -> float:
    pred_counts = _token_counts(prediction)
    ref_counts = _token_counts(reference)
    if not pred_counts and not ref_counts:
        return 1.0
    if not pred_counts or not ref_counts:
        return 0.0

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    if pred_total <= 0 or ref_total <= 0 or overlap <= 0:
        return 0.0

    precision = overlap / float(pred_total)
    recall = overlap / float(ref_total)
    if precision + recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _quality_scores(prediction: str, reference: str) -> dict[str, Any]:
    normalized_prediction = _normalize_for_quality(prediction)
    normalized_reference = _normalize_for_quality(reference)
    exact_match = normalized_prediction == normalized_reference and bool(normalized_reference)
    contains_reference = bool(normalized_reference) and normalized_reference in normalized_prediction
    token_f1 = _token_f1(prediction, reference)
    quality_score = (0.4 if exact_match else 0.0) + (0.3 if contains_reference else 0.0) + (0.3 * token_f1)
    return {
        "exact_match": exact_match,
        "contains_reference": contains_reference,
        "token_f1": token_f1,
        "quality_score": quality_score,
    }


def _truncate_text(text: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _build_disaggregated_components(
    *,
    model_name: str,
    serving_config: dict[str, Any],
    max_model_len: int,
    allow_simulated_backend: bool,
) -> tuple[VLLMBackendConfig, VLLMBackendConfig, DisaggregatedScheduler, dict[str, Any]]:
    mode = str(serving_config.get("mode", "disaggregated")).strip().lower()
    backend_name = str(serving_config.get("backend", "vllm")).strip().lower()
    if backend_name != "vllm":
        raise ValueError(f"Unsupported backend '{backend_name}'. Expected 'vllm'.")
    if mode != "disaggregated":
        raise ValueError(f"Expected disaggregated serving config for disaggregated mode, got '{mode}'.")

    gpu_assignment = core._resolve_disaggregated_gpu_assignment(serving_config)
    prefill_config = core._build_vllm_backend_config(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=max_model_len,
        allow_simulated_backend=allow_simulated_backend,
        visible_devices=core._safe_optional_str(gpu_assignment.get("prefill_visible_devices")),
        device=core._safe_optional_str(serving_config.get("prefill_device")),
    )
    decode_config = core._build_vllm_backend_config(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=max_model_len,
        allow_simulated_backend=allow_simulated_backend,
        visible_devices=core._safe_optional_str(gpu_assignment.get("decode_visible_devices")),
        device=core._safe_optional_str(serving_config.get("decode_device")),
    )
    scheduler = DisaggregatedScheduler(
        prefill_pool=core._safe_str(serving_config.get("prefill_pool_name"), "prefill"),
        prefill_pool_size=core._safe_int(serving_config.get("prefill_pool_size"), 1),
        decode_pool=core._safe_str(serving_config.get("decode_pool_name"), "decode"),
        decode_pool_size=core._safe_int(serving_config.get("decode_pool_size"), 1),
        broker=LocalQueuePDBroker(
            bytes_per_token=core._safe_int(serving_config.get("bytes_per_token"), 2048)
        ),
    )
    return (prefill_config, decode_config, scheduler, gpu_assignment)


def _empty_gpu_assignment() -> dict[str, Any]:
    return {
        "strategy": "not_applicable",
        "warning": None,
        "tensor_parallel_size": None,
        "available_gpu_indices": [],
        "prefill_visible_devices": None,
        "decode_visible_devices": None,
    }


def _build_request_for_sample(
    *,
    request_id: str,
    sample: BenchmarkSample,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    experiment_name: str,
) -> GenerationRequest:
    return GenerationRequest(
        prompt=sample.prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        request_id=request_id,
        metadata={
            "experiment_name": experiment_name,
            "benchmark_task": sample.task,
            "sample_id": sample.sample_id,
            "seed": seed,
        },
    )


def _execute_e3_mode(
    *,
    mode: str,
    model_name: str,
    serving_config: dict[str, Any],
    experiment_name: str,
    concurrency_levels: list[int],
    waves_per_level: int,
    prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    warmup: bool,
    strict_backend: bool,
    allow_simulated_backend: bool,
) -> dict[str, Any]:
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=core._safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode(mode)

    started_at = core._utc_now()
    wall_start = perf_counter()
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    backend_modes: dict[str, str] = {}
    request_records: list[dict[str, Any]] = []
    concurrency_results: list[dict[str, Any]] = []
    route_log: list[dict[str, Any]] = []
    broker_snapshot: dict[str, Any] | None = None
    broker_completed_handoffs: list[dict[str, Any]] = []
    gpu_assignment = _empty_gpu_assignment()

    scheduler: AggregatedScheduler | DisaggregatedScheduler | None = None
    aggregated_backend: VLLMBackend | None = None
    prefill_backend: VLLMBackend | None = None
    decode_backend: VLLMBackend | None = None

    success = False
    error_message: str | None = None
    try:
        if mode == "aggregated":
            backend_config, scheduler_obj = core._build_aggregated_components(
                model_name=model_name,
                serving_config=serving_config,
                max_model_len=prompt_tokens + max_new_tokens,
                allow_simulated_backend=allow_simulated_backend,
            )
            scheduler = scheduler_obj
            aggregated_backend = VLLMBackend(backend_config)
            aggregated_backend.start()
            backend_modes["aggregated"] = aggregated_backend.mode
            if warmup:
                aggregated_backend.warmup()
        elif mode == "disaggregated":
            (
                prefill_config,
                decode_config,
                scheduler_obj,
                resolved_gpu_assignment,
            ) = _build_disaggregated_components(
                model_name=model_name,
                serving_config=serving_config,
                max_model_len=prompt_tokens + max_new_tokens,
                allow_simulated_backend=allow_simulated_backend,
            )
            scheduler = scheduler_obj
            gpu_assignment = resolved_gpu_assignment
            prefill_backend = VLLMBackend(prefill_config)
            decode_backend = VLLMBackend(decode_config)
            prefill_backend.start()
            decode_backend.start()
            backend_modes["prefill"] = prefill_backend.mode
            backend_modes["decode"] = decode_backend.mode
            if warmup:
                prefill_backend.warmup()
                decode_backend.warmup()
        else:
            raise ValueError(f"Unsupported mode '{mode}' for E3 execution.")

        for concurrency in concurrency_levels:
            total_wave_wall_ms = 0.0
            total_completion_tokens = 0
            total_requests = 0
            goodput_request_count = 0
            level_ttft_values: list[float] = []
            level_tpot_values: list[float] = []

            for wave_idx in range(waves_per_level):
                requests: list[GenerationRequest] = []
                for slot in range(concurrency):
                    request_id = (
                        f"{experiment_name}-{mode}-c{concurrency}-w{wave_idx + 1:03d}-r{slot + 1:03d}"
                    )
                    requests.append(
                        core._build_generation_request(
                            request_id=request_id,
                            prompt_tokens=prompt_tokens,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            seed=seed,
                            experiment_name=experiment_name,
                        )
                    )

                wave_started = perf_counter()
                if mode == "aggregated":
                    assert aggregated_backend is not None
                    assert isinstance(scheduler, AggregatedScheduler)
                    prefill_by_request: dict[str, dict[str, Any]] = {}
                    for request in requests:
                        scheduler.route_prefill(request)
                        prefill_by_request[request.request_id or ""] = aggregated_backend.prefill(request)
                        scheduler.route_decode(request.request_id or "")
                    if hasattr(aggregated_backend, "generate_batch"):
                        generation_results = aggregated_backend.generate_batch(requests)
                    else:
                        generation_results = [aggregated_backend.generate(request) for request in requests]
                    wave_records = [
                        (
                            request,
                            prefill_by_request.get(request.request_id or "", {}),
                            generation_result,
                            core._safe_str(serving_config.get("pool_name"), "aggregated"),
                            core._safe_str(serving_config.get("pool_name"), "aggregated"),
                        )
                        for request, generation_result in zip(requests, generation_results)
                    ]
                else:
                    assert prefill_backend is not None
                    assert decode_backend is not None
                    assert isinstance(scheduler, DisaggregatedScheduler)
                    wave_records = []
                    for request in requests:
                        prefill_pool = scheduler.route_prefill(request)
                        prefill_result = prefill_backend.prefill(request)
                        decode_pool = scheduler.route_decode(request.request_id or "")
                        generation_result = decode_backend.generate(request)
                        wave_records.append(
                            (request, prefill_result, generation_result, prefill_pool, decode_pool)
                        )

                wave_wall_ms = (perf_counter() - wave_started) * 1000.0
                total_wave_wall_ms += wave_wall_ms

                for request, prefill_result, generation_result, prefill_pool, decode_pool in wave_records:
                    total_requests += 1
                    total_completion_tokens += generation_result.completion_tokens
                    level_ttft_values.append(generation_result.ttft_ms)
                    level_tpot_values.append(generation_result.tpot_ms)
                    if (
                        generation_result.ttft_ms <= ttft_slo_ms
                        and generation_result.tpot_ms <= tpot_slo_ms
                    ):
                        goodput_request_count += 1
                    latency_collector.record_sample(
                        request_id=request.request_id or f"{mode}-unknown-request",
                        ttft_ms=generation_result.ttft_ms,
                        tpot_ms=generation_result.tpot_ms,
                        prompt_tokens=generation_result.prompt_tokens,
                        completion_tokens=generation_result.completion_tokens,
                        metadata={
                            **generation_result.metadata,
                            "mode": mode,
                            "concurrency_level": concurrency,
                            "wave_index": wave_idx + 1,
                        },
                    )
                    request_records.append(
                        {
                            "request_id": request.request_id,
                            "mode": mode,
                            "concurrency": concurrency,
                            "wave_index": wave_idx + 1,
                            "prompt_tokens_observed": generation_result.prompt_tokens,
                            "completion_tokens": generation_result.completion_tokens,
                            "ttft_ms": generation_result.ttft_ms,
                            "tpot_ms": generation_result.tpot_ms,
                            "prefill_ms": core._safe_float(prefill_result.get("prefill_ms"), 0.0),
                            "prefill_pool": prefill_pool,
                            "decode_pool": decode_pool,
                        }
                    )

            level_wall_seconds = total_wave_wall_ms / 1000.0 if total_wave_wall_ms > 0 else 0.0
            throughput_tokens_per_s = (
                (total_completion_tokens / level_wall_seconds) if level_wall_seconds > 0 else 0.0
            )
            goodput_rps = (
                (goodput_request_count / level_wall_seconds) if level_wall_seconds > 0 else 0.0
            )
            concurrency_results.append(
                {
                    "mode": mode,
                    "concurrency": concurrency,
                    "waves": waves_per_level,
                    "request_count": total_requests,
                    "total_completion_tokens": total_completion_tokens,
                    "total_wall_ms": total_wave_wall_ms,
                    "throughput_tokens_per_s": throughput_tokens_per_s,
                    "goodput_rps": goodput_rps,
                    "goodput_request_count": goodput_request_count,
                    "ttft_ms": core._distribution(level_ttft_values),
                    "tpot_ms": core._distribution(level_tpot_values),
                }
            )

        if strict_backend and any(value != "vllm" for value in backend_modes.values()):
            raise RuntimeError("Strict backend mode requires real vLLM execution.")

        success = True
    except Exception as exc:
        error_message = str(exc)
    finally:
        if aggregated_backend is not None:
            aggregated_backend.stop()
        if prefill_backend is not None:
            prefill_backend.stop()
        if decode_backend is not None:
            decode_backend.stop()
        latency_collector.stop()
        gpu_collector.stop()
        kv_transfer_collector.stop()

    wall_time_ms = (perf_counter() - wall_start) * 1000.0
    finished_at = core._utc_now()
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)
    if isinstance(scheduler, DisaggregatedScheduler):
        route_log = scheduler.route_log()
        broker_snapshot = scheduler.broker.snapshot()
        broker_completed_handoffs = scheduler.broker.completed_handoffs()
        kv_transfer_collector.ingest_completed_handoffs(broker_completed_handoffs)
    elif isinstance(scheduler, AggregatedScheduler):
        route_log = scheduler.route_log()

    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }
    return {
        "mode": mode,
        "success": success,
        "error": error_message,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ms": wall_time_ms,
        "backend_modes": backend_modes,
        "route_log": route_log,
        "request_records": request_records,
        "concurrency_results": concurrency_results,
        "profiling": profiling_snapshot,
        "gpu_assignment": gpu_assignment,
        "broker_snapshot": broker_snapshot,
        "broker_completed_handoffs": broker_completed_handoffs,
    }


def _results_by_concurrency(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for result in results:
        concurrency = core._safe_int(result.get("concurrency"), -1)
        if concurrency >= 0:
            index[concurrency] = result
    return index


def run_e3(
    *,
    experiment_config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
    aggregated_serving_config_path: str | Path | None = None,
    disaggregated_serving_config_path: str | Path | None = None,
) -> E3RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    experiment_name = str(experiment_config.get("name", "e3_disaggregated_vs_aggregated")).strip()
    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")
    if "seed" not in experiment_config:
        raise ValueError("E3 requires a fixed integer seed in experiment config.")
    seed = core._safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")

    concurrency_levels = core._safe_int_list(
        experiment_config.get("concurrency_levels"),
        key_name="concurrency_levels",
        min_value=1,
    )
    waves_per_level = core._safe_int(experiment_config.get("waves_per_level"), 3)
    if waves_per_level < 1:
        raise ValueError("E3 field 'waves_per_level' must be >= 1.")
    prompt_tokens = core._safe_int(experiment_config.get("prompt_tokens"), 4096)
    max_new_tokens = core._safe_int(experiment_config.get("max_new_tokens"), 64)
    if prompt_tokens < 8:
        raise ValueError("E3 field 'prompt_tokens' must be >= 8.")
    if max_new_tokens < 1:
        raise ValueError("E3 field 'max_new_tokens' must be >= 1.")
    temperature = core._safe_float(experiment_config.get("temperature"), 0.0)
    top_p = core._safe_float(experiment_config.get("top_p"), 1.0)
    ttft_slo_ms = core._safe_float(experiment_config.get("ttft_slo_ms"), 4000.0)
    tpot_slo_ms = core._safe_float(experiment_config.get("tpot_slo_ms"), 120.0)

    configured_agg_path = core._safe_str(
        experiment_config.get("aggregated_serving_config"),
        "configs/serving/aggregated.yaml",
    )
    configured_disagg_path = core._safe_str(
        experiment_config.get("disaggregated_serving_config"),
        "configs/serving/disaggregated.yaml",
    )
    agg_serving_path = str(Path(aggregated_serving_config_path or configured_agg_path))
    disagg_serving_path = str(Path(disaggregated_serving_config_path or configured_disagg_path))
    aggregated_serving_config = load_simple_yaml(agg_serving_path)
    disaggregated_serving_config = load_simple_yaml(disagg_serving_path)

    aggregated_allow_simulated = (
        allow_simulated_backend
        if allow_simulated_backend is not None
        else core._safe_bool(aggregated_serving_config.get("allow_simulated"), False)
    )
    disaggregated_allow_simulated = (
        allow_simulated_backend
        if allow_simulated_backend is not None
        else core._safe_bool(disaggregated_serving_config.get("allow_simulated"), False)
    )
    if strict_backend:
        aggregated_allow_simulated = False
        disaggregated_allow_simulated = False

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": experiment_name,
            "serving_mode": "comparison",
            "aggregated_serving_config_path": str(Path(agg_serving_path).resolve()),
            "disaggregated_serving_config_path": str(Path(disagg_serving_path).resolve()),
            "seed": seed,
            "strict_backend": strict_backend,
            "concurrency_levels": concurrency_levels,
            "waves_per_level": waves_per_level,
            **core._as_mapping(experiment_config.get("notes", {}), "notes"),
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at = core._utc_now()
    run_wall_start = perf_counter()
    aggregated_result = _execute_e3_mode(
        mode="aggregated",
        model_name=model_name,
        serving_config=aggregated_serving_config,
        experiment_name=experiment_name,
        concurrency_levels=concurrency_levels,
        waves_per_level=waves_per_level,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        warmup=warmup,
        strict_backend=strict_backend,
        allow_simulated_backend=aggregated_allow_simulated,
    )
    disaggregated_result = _execute_e3_mode(
        mode="disaggregated",
        model_name=model_name,
        serving_config=disaggregated_serving_config,
        experiment_name=experiment_name,
        concurrency_levels=concurrency_levels,
        waves_per_level=waves_per_level,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        warmup=warmup,
        strict_backend=strict_backend,
        allow_simulated_backend=disaggregated_allow_simulated,
    )
    wall_time_ms = (perf_counter() - run_wall_start) * 1000.0
    finished_at = core._utc_now()

    agg_map = _results_by_concurrency(aggregated_result["concurrency_results"])
    disagg_map = _results_by_concurrency(disaggregated_result["concurrency_results"])
    bytes_per_token = core._safe_int(disaggregated_serving_config.get("bytes_per_token"), 2048)
    disagg_kv_summary = (
        disaggregated_result.get("profiling", {})
        .get("kv_transfer", {})
        .get("summary", {})
    )
    disagg_stall_ratio = _to_optional_float(disagg_kv_summary.get("stall_ratio"))

    comparison_rows: list[dict[str, Any]] = []
    for concurrency in sorted(set(agg_map.keys()).union(disagg_map.keys())):
        agg_row = agg_map.get(concurrency, {})
        disagg_row = disagg_map.get(concurrency, {})
        agg_ttft_p95 = _to_optional_float((agg_row.get("ttft_ms") or {}).get("p95"))
        agg_tpot_p95 = _to_optional_float((agg_row.get("tpot_ms") or {}).get("p95"))
        disagg_ttft_p95 = _to_optional_float((disagg_row.get("ttft_ms") or {}).get("p95"))
        disagg_tpot_p95 = _to_optional_float((disagg_row.get("tpot_ms") or {}).get("p95"))
        agg_goodput = _to_optional_float(agg_row.get("goodput_rps"))
        disagg_goodput = _to_optional_float(disagg_row.get("goodput_rps"))
        agg_throughput = _to_optional_float(agg_row.get("throughput_tokens_per_s"))
        disagg_throughput = _to_optional_float(disagg_row.get("throughput_tokens_per_s"))
        disagg_request_count = core._safe_int(disagg_row.get("request_count"), 0)
        comparison_rows.append(
            {
                "concurrency": concurrency,
                "aggregated_ttft_p95_ms": agg_ttft_p95,
                "disaggregated_ttft_p95_ms": disagg_ttft_p95,
                "delta_ttft_p95_ms": _safe_delta(disagg_ttft_p95, agg_ttft_p95),
                "aggregated_tpot_p95_ms": agg_tpot_p95,
                "disaggregated_tpot_p95_ms": disagg_tpot_p95,
                "delta_tpot_p95_ms": _safe_delta(disagg_tpot_p95, agg_tpot_p95),
                "aggregated_goodput_rps": agg_goodput,
                "disaggregated_goodput_rps": disagg_goodput,
                "delta_goodput_rps": _safe_delta(disagg_goodput, agg_goodput),
                "aggregated_throughput_tokens_per_s": agg_throughput,
                "disaggregated_throughput_tokens_per_s": disagg_throughput,
                "delta_throughput_tokens_per_s": _safe_delta(disagg_throughput, agg_throughput),
                "disaggregated_transfer_estimated_bytes": disagg_request_count
                * prompt_tokens
                * bytes_per_token,
                "disaggregated_transfer_stall_ratio": disagg_stall_ratio,
            }
        )

    e3_csv_path = artifacts_dir / "e3_mode_comparison.csv"
    core._write_csv_rows(
        output_path=e3_csv_path,
        fieldnames=[
            "concurrency",
            "aggregated_ttft_p95_ms",
            "disaggregated_ttft_p95_ms",
            "delta_ttft_p95_ms",
            "aggregated_tpot_p95_ms",
            "disaggregated_tpot_p95_ms",
            "delta_tpot_p95_ms",
            "aggregated_goodput_rps",
            "disaggregated_goodput_rps",
            "delta_goodput_rps",
            "aggregated_throughput_tokens_per_s",
            "disaggregated_throughput_tokens_per_s",
            "delta_throughput_tokens_per_s",
            "disaggregated_transfer_estimated_bytes",
            "disaggregated_transfer_stall_ratio",
        ],
        rows=comparison_rows,
    )

    plot_x = [float(row["concurrency"]) for row in comparison_rows]
    e3_latency_plot_path = artifacts_dir / "e3_latency_comparison.svg"
    core._write_line_chart_svg(
        output_path=e3_latency_plot_path,
        title=f"{experiment_name}: TTFT p95 Comparison",
        x_label="Concurrency Level",
        y_label="TTFT p95 (ms)",
        x_values=plot_x,
        series=[
            (
                "Aggregated",
                [_to_plot_value(row["aggregated_ttft_p95_ms"]) for row in comparison_rows],
            ),
            (
                "Disaggregated",
                [_to_plot_value(row["disaggregated_ttft_p95_ms"]) for row in comparison_rows],
            ),
        ],
    )
    e3_goodput_plot_path = artifacts_dir / "e3_goodput_comparison.svg"
    core._write_line_chart_svg(
        output_path=e3_goodput_plot_path,
        title=f"{experiment_name}: Goodput Comparison",
        x_label="Concurrency Level",
        y_label="Goodput (req/s)",
        x_values=plot_x,
        series=[
            (
                "Aggregated",
                [_to_plot_value(row["aggregated_goodput_rps"]) for row in comparison_rows],
            ),
            (
                "Disaggregated",
                [_to_plot_value(row["disaggregated_goodput_rps"]) for row in comparison_rows],
            ),
        ],
    )

    success = aggregated_result["success"] and disaggregated_result["success"]
    error_messages = [
        message
        for message in [aggregated_result.get("error"), disaggregated_result.get("error")]
        if isinstance(message, str) and message.strip()
    ]
    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "aggregated_success": aggregated_result["success"],
        "disaggregated_success": disaggregated_result["success"],
        "comparison_rows_generated": len(comparison_rows),
        "plots_generated": e3_latency_plot_path.exists() and e3_goodput_plot_path.exists(),
        "environment_manifest_present": True,
    }
    summary_payload: dict[str, Any] = {
        "run_id": metadata.run_id,
        "experiment_name": experiment_name,
        "success": success,
        "error": "; ".join(error_messages) if error_messages else None,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ms": wall_time_ms,
        "mode": "comparison",
        "backend": "vllm",
        "checks": checks,
        "seed": seed,
        "slo": {
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
        },
        "experiment_config_path": str(Path(experiment_config_path).resolve()),
        "aggregated_serving_config_path": str(Path(agg_serving_path).resolve()),
        "disaggregated_serving_config_path": str(Path(disagg_serving_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "comparison_rows": comparison_rows,
        "mode_runs": {
            "aggregated": aggregated_result,
            "disaggregated": disaggregated_result,
        },
        "artifacts": {
            "e3_mode_comparison_csv": str(e3_csv_path.resolve()),
            "e3_latency_comparison_svg": str(e3_latency_plot_path.resolve()),
            "e3_goodput_comparison_svg": str(e3_goodput_plot_path.resolve()),
        },
        "environment_manifest": core._env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }
    summary_path = run_dir / "e3_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return E3RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode="comparison",
        success=success,
    )


def run_e4(
    *,
    experiment_config_path: str | Path,
    serving_config_path: str | Path = "configs/serving/aggregated.yaml",
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
    benchmark_config_path: str | Path | None = None,
) -> E4RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    serving_config = load_simple_yaml(serving_config_path)
    experiment_name = str(experiment_config.get("name", "e4_capability_subset")).strip()

    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")
    if "seed" not in experiment_config:
        raise ValueError("E4 requires a fixed integer seed in experiment config.")
    seed = core._safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")
    max_new_tokens = core._safe_int(experiment_config.get("max_new_tokens"), 64)
    temperature = core._safe_float(experiment_config.get("temperature"), 0.0)
    top_p = core._safe_float(experiment_config.get("top_p"), 1.0)
    if max_new_tokens < 1:
        raise ValueError("E4 field 'max_new_tokens' must be >= 1.")

    benchmark_path = str(
        Path(
            benchmark_config_path
            or core._safe_str(experiment_config.get("benchmark_config"), "configs/benchmarks/wp1_core.yaml")
        )
    )
    benchmark_suite = load_benchmark_suite(benchmark_path)

    if allow_simulated_backend is None:
        allow_simulated_backend = core._safe_bool(serving_config.get("allow_simulated"), False)
    if strict_backend:
        allow_simulated_backend = False

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": experiment_name,
            "serving_mode": "aggregated",
            "serving_config_path": str(Path(serving_config_path).resolve()),
            "benchmark_config_path": str(Path(benchmark_path).resolve()),
            "seed": seed,
            "strict_backend": strict_backend,
            **core._as_mapping(experiment_config.get("notes", {}), "notes"),
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_tokens = max(
        [len(sample.prompt.split()) for task in benchmark_suite.tasks for sample in task.samples] or [64]
    )
    backend_config, scheduler = core._build_aggregated_components(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=max_prompt_tokens + max_new_tokens,
        allow_simulated_backend=allow_simulated_backend,
    )

    started_at = core._utc_now()
    wall_start = perf_counter()
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=core._safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode("aggregated")
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    backend_modes: dict[str, str] = {}
    sample_records: list[dict[str, Any]] = []
    success = False
    error_message: str | None = None
    backend: VLLMBackend | None = None
    try:
        backend = VLLMBackend(backend_config)
        backend.start()
        backend_modes["aggregated"] = backend.mode
        if warmup:
            backend.warmup()

        for task in benchmark_suite.tasks:
            for sample_index, sample in enumerate(task.samples):
                request_id = f"{experiment_name}-{task.name}-{sample_index + 1:03d}"
                request = _build_request_for_sample(
                    request_id=request_id,
                    sample=sample,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    experiment_name=experiment_name,
                )
                _, _, prefill_result, generation_result = core._run_single_aggregated_request(
                    backend=backend,
                    scheduler=scheduler,
                    request=request,
                )
                quality = _quality_scores(generation_result.text, sample.reference)
                latency_collector.record_sample(
                    request_id=request_id,
                    ttft_ms=generation_result.ttft_ms,
                    tpot_ms=generation_result.tpot_ms,
                    prompt_tokens=generation_result.prompt_tokens,
                    completion_tokens=generation_result.completion_tokens,
                    metadata={
                        **generation_result.metadata,
                        "benchmark_task": task.name,
                        "sample_id": sample.sample_id,
                    },
                )
                sample_records.append(
                    {
                        "task": task.name,
                        "task_source": task.source,
                        "sample_id": sample.sample_id,
                        "reference": sample.reference,
                        "output_text": _truncate_text(generation_result.text),
                        "ttft_ms": generation_result.ttft_ms,
                        "tpot_ms": generation_result.tpot_ms,
                        "prompt_tokens_observed": generation_result.prompt_tokens,
                        "completion_tokens": generation_result.completion_tokens,
                        "prefill_ms": core._safe_float(prefill_result.get("prefill_ms"), 0.0),
                        "exact_match": quality["exact_match"],
                        "contains_reference": quality["contains_reference"],
                        "token_f1": quality["token_f1"],
                        "quality_score": quality["quality_score"],
                    }
                )

        if strict_backend and any(value != "vllm" for value in backend_modes.values()):
            raise RuntimeError("Strict backend mode requires real vLLM execution.")
        success = True
    except Exception as exc:
        error_message = str(exc)
    finally:
        if backend is not None:
            backend.stop()
        latency_collector.stop()
        gpu_collector.stop()
        kv_transfer_collector.stop()

    wall_time_ms = (perf_counter() - wall_start) * 1000.0
    finished_at = core._utc_now()
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)
    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }

    quality_by_task: list[dict[str, Any]] = []
    for task in benchmark_suite.tasks:
        task_records = [record for record in sample_records if record["task"] == task.name]
        ttft_values = [float(record["ttft_ms"]) for record in task_records]
        tpot_values = [float(record["tpot_ms"]) for record in task_records]
        token_f1_values = [float(record["token_f1"]) for record in task_records]
        quality_values = [float(record["quality_score"]) for record in task_records]
        exact_matches = [1.0 if bool(record["exact_match"]) else 0.0 for record in task_records]
        contains_reference = [
            1.0 if bool(record["contains_reference"]) else 0.0 for record in task_records
        ]
        quality_by_task.append(
            {
                "task": task.name,
                "source": task.source,
                "sample_count": len(task_records),
                "exact_match_rate": _safe_average(exact_matches),
                "contains_reference_rate": _safe_average(contains_reference),
                "token_f1_avg": _safe_average(token_f1_values),
                "quality_score_avg": _safe_average(quality_values),
                "ttft_ms": core._distribution(ttft_values),
                "tpot_ms": core._distribution(tpot_values),
            }
        )

    overall_exact = [1.0 if bool(record["exact_match"]) else 0.0 for record in sample_records]
    overall_contains = [
        1.0 if bool(record["contains_reference"]) else 0.0 for record in sample_records
    ]
    overall_token_f1 = [float(record["token_f1"]) for record in sample_records]
    overall_quality = [float(record["quality_score"]) for record in sample_records]
    overall_metrics = {
        "sample_count": len(sample_records),
        "exact_match_rate": _safe_average(overall_exact),
        "contains_reference_rate": _safe_average(overall_contains),
        "token_f1_avg": _safe_average(overall_token_f1),
        "quality_score_avg": _safe_average(overall_quality),
    }

    e4_task_csv_path = artifacts_dir / "e4_quality_by_task.csv"
    core._write_csv_rows(
        output_path=e4_task_csv_path,
        fieldnames=[
            "task",
            "source",
            "sample_count",
            "exact_match_rate",
            "contains_reference_rate",
            "token_f1_avg",
            "quality_score_avg",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "tpot_p50_ms",
            "tpot_p95_ms",
        ],
        rows=[
            {
                "task": row["task"],
                "source": row["source"],
                "sample_count": row["sample_count"],
                "exact_match_rate": row["exact_match_rate"],
                "contains_reference_rate": row["contains_reference_rate"],
                "token_f1_avg": row["token_f1_avg"],
                "quality_score_avg": row["quality_score_avg"],
                "ttft_p50_ms": (row["ttft_ms"] or {}).get("p50"),
                "ttft_p95_ms": (row["ttft_ms"] or {}).get("p95"),
                "tpot_p50_ms": (row["tpot_ms"] or {}).get("p50"),
                "tpot_p95_ms": (row["tpot_ms"] or {}).get("p95"),
            }
            for row in quality_by_task
        ],
    )
    e4_sample_csv_path = artifacts_dir / "e4_sample_scores.csv"
    core._write_csv_rows(
        output_path=e4_sample_csv_path,
        fieldnames=[
            "task",
            "task_source",
            "sample_id",
            "reference",
            "output_text",
            "ttft_ms",
            "tpot_ms",
            "prompt_tokens_observed",
            "completion_tokens",
            "prefill_ms",
            "exact_match",
            "contains_reference",
            "token_f1",
            "quality_score",
        ],
        rows=sample_records,
    )

    task_labels = [row["task"] for row in quality_by_task]
    e4_quality_plot_path = artifacts_dir / "e4_quality_by_task.svg"
    core._write_line_chart_svg(
        output_path=e4_quality_plot_path,
        title=f"{experiment_name}: Quality Score by Task",
        x_label="Task Index",
        y_label="Quality Score",
        x_values=[float(index + 1) for index in range(len(task_labels))],
        series=[
            ("Quality score", [_to_plot_value(row["quality_score_avg"]) for row in quality_by_task]),
            ("Token F1", [_to_plot_value(row["token_f1_avg"]) for row in quality_by_task]),
        ],
    )

    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "all_backends_real_vllm": bool(backend_modes)
        and all(mode_name == "vllm" for mode_name in backend_modes.values()),
        "executed_samples": len(sample_records),
        "environment_manifest_present": True,
        "task_csv_generated": e4_task_csv_path.exists(),
    }
    summary_payload: dict[str, Any] = {
        "run_id": metadata.run_id,
        "experiment_name": experiment_name,
        "success": success,
        "error": error_message,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ms": wall_time_ms,
        "mode": "aggregated",
        "backend": "vllm",
        "backend_modes": backend_modes,
        "seed": seed,
        "checks": checks,
        "experiment_config_path": str(Path(experiment_config_path).resolve()),
        "serving_config_path": str(Path(serving_config_path).resolve()),
        "benchmark_config_path": str(Path(benchmark_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "profiling": profiling_snapshot,
        "route_log": scheduler.route_log(),
        "quality_by_task": quality_by_task,
        "quality_overall": overall_metrics,
        "sample_records": sample_records,
        "benchmark_suite": {
            "name": benchmark_suite.name,
            "config_path": benchmark_suite.config_path,
            "seed": benchmark_suite.seed,
            "sample_count": benchmark_suite.sample_count,
            "tasks": [
                {
                    "name": task.name,
                    "source": task.source,
                    "sample_count": len(task.samples),
                    "sample_ids": [sample.sample_id for sample in task.samples],
                }
                for task in benchmark_suite.tasks
            ],
        },
        "artifacts": {
            "e4_quality_by_task_csv": str(e4_task_csv_path.resolve()),
            "e4_sample_scores_csv": str(e4_sample_csv_path.resolve()),
            "e4_quality_by_task_svg": str(e4_quality_plot_path.resolve()),
        },
        "environment_manifest": core._env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }
    summary_path = run_dir / "e4_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return E4RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode="aggregated",
        success=success,
    )


def _run_e5_memory_case(
    *,
    mode: str,
    model_name: str,
    serving_config: dict[str, Any],
    experiment_name: str,
    prompt_tokens: int,
    max_new_tokens: int,
    trials_per_length: int,
    temperature: float,
    top_p: float,
    seed: int,
    warmup: bool,
    strict_backend: bool,
    allow_simulated_backend: bool,
) -> dict[str, Any]:
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=core._safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode(mode)
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    backend_modes: dict[str, str] = {}
    request_records: list[dict[str, Any]] = []
    route_log: list[dict[str, Any]] = []
    broker_snapshot: dict[str, Any] | None = None
    broker_completed_handoffs: list[dict[str, Any]] = []
    gpu_assignment = _empty_gpu_assignment()

    scheduler: AggregatedScheduler | DisaggregatedScheduler | None = None
    aggregated_backend: VLLMBackend | None = None
    prefill_backend: VLLMBackend | None = None
    decode_backend: VLLMBackend | None = None

    wall_start = perf_counter()
    success = False
    error_message: str | None = None
    try:
        if mode == "aggregated":
            backend_config, scheduler_obj = core._build_aggregated_components(
                model_name=model_name,
                serving_config=serving_config,
                max_model_len=prompt_tokens + max_new_tokens,
                allow_simulated_backend=allow_simulated_backend,
            )
            scheduler = scheduler_obj
            aggregated_backend = VLLMBackend(backend_config)
            aggregated_backend.start()
            backend_modes["aggregated"] = aggregated_backend.mode
            if warmup:
                aggregated_backend.warmup()
        elif mode == "disaggregated":
            (
                prefill_config,
                decode_config,
                scheduler_obj,
                resolved_gpu_assignment,
            ) = _build_disaggregated_components(
                model_name=model_name,
                serving_config=serving_config,
                max_model_len=prompt_tokens + max_new_tokens,
                allow_simulated_backend=allow_simulated_backend,
            )
            scheduler = scheduler_obj
            gpu_assignment = resolved_gpu_assignment
            prefill_backend = VLLMBackend(prefill_config)
            decode_backend = VLLMBackend(decode_config)
            prefill_backend.start()
            decode_backend.start()
            backend_modes["prefill"] = prefill_backend.mode
            backend_modes["decode"] = decode_backend.mode
            if warmup:
                prefill_backend.warmup()
                decode_backend.warmup()
        else:
            raise ValueError(f"Unsupported mode '{mode}' for E5.")

        for trial_idx in range(trials_per_length):
            request = core._build_generation_request(
                request_id=f"{experiment_name}-{mode}-pt{prompt_tokens}-trial{trial_idx + 1:03d}",
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                experiment_name=experiment_name,
            )
            if mode == "aggregated":
                assert aggregated_backend is not None
                assert isinstance(scheduler, AggregatedScheduler)
                prefill_pool, decode_pool, prefill_result, generation_result = (
                    core._run_single_aggregated_request(
                        backend=aggregated_backend,
                        scheduler=scheduler,
                        request=request,
                    )
                )
            else:
                assert prefill_backend is not None
                assert decode_backend is not None
                assert isinstance(scheduler, DisaggregatedScheduler)
                prefill_pool = scheduler.route_prefill(request)
                prefill_result = prefill_backend.prefill(request)
                decode_pool = scheduler.route_decode(request.request_id or "")
                generation_result = decode_backend.generate(request)

            latency_collector.record_sample(
                request_id=request.request_id or f"{mode}-memory-unknown-request",
                ttft_ms=generation_result.ttft_ms,
                tpot_ms=generation_result.tpot_ms,
                prompt_tokens=generation_result.prompt_tokens,
                completion_tokens=generation_result.completion_tokens,
                metadata={
                    **generation_result.metadata,
                    "mode": mode,
                    "prompt_tokens_target": prompt_tokens,
                    "trial_index": trial_idx + 1,
                },
            )
            request_records.append(
                {
                    "request_id": request.request_id,
                    "mode": mode,
                    "trial_index": trial_idx + 1,
                    "prompt_tokens_target": prompt_tokens,
                    "prompt_tokens_observed": generation_result.prompt_tokens,
                    "completion_tokens": generation_result.completion_tokens,
                    "ttft_ms": generation_result.ttft_ms,
                    "tpot_ms": generation_result.tpot_ms,
                    "prefill_ms": core._safe_float(prefill_result.get("prefill_ms"), 0.0),
                    "prefill_pool": prefill_pool,
                    "decode_pool": decode_pool,
                }
            )

        if strict_backend and any(value != "vllm" for value in backend_modes.values()):
            raise RuntimeError("Strict backend mode requires real vLLM execution.")
        success = True
    except Exception as exc:
        error_message = str(exc)
    finally:
        if aggregated_backend is not None:
            aggregated_backend.stop()
        if prefill_backend is not None:
            prefill_backend.stop()
        if decode_backend is not None:
            decode_backend.stop()
        latency_collector.stop()
        gpu_collector.stop()
        kv_transfer_collector.stop()

    wall_time_ms = (perf_counter() - wall_start) * 1000.0
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)
    if isinstance(scheduler, DisaggregatedScheduler):
        route_log = scheduler.route_log()
        broker_snapshot = scheduler.broker.snapshot()
        broker_completed_handoffs = scheduler.broker.completed_handoffs()
        kv_transfer_collector.ingest_completed_handoffs(broker_completed_handoffs)
    elif isinstance(scheduler, AggregatedScheduler):
        route_log = scheduler.route_log()

    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }
    return {
        "mode": mode,
        "success": success,
        "error": error_message,
        "wall_time_ms": wall_time_ms,
        "backend_modes": backend_modes,
        "gpu_assignment": gpu_assignment,
        "route_log": route_log,
        "request_records": request_records,
        "profiling": profiling_snapshot,
        "broker_snapshot": broker_snapshot,
        "broker_completed_handoffs": broker_completed_handoffs,
    }


def run_e5(
    *,
    experiment_config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
    aggregated_serving_config_path: str | Path | None = None,
    disaggregated_serving_config_path: str | Path | None = None,
) -> E5RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    experiment_name = str(experiment_config.get("name", "e5_memory_decomposition")).strip()
    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")
    if "seed" not in experiment_config:
        raise ValueError("E5 requires a fixed integer seed in experiment config.")
    seed = core._safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")

    modes = _safe_mode_list(experiment_config.get("modes"), ["aggregated", "disaggregated"])
    supported_modes = {"aggregated", "disaggregated"}
    invalid_modes = [mode for mode in modes if mode not in supported_modes]
    if invalid_modes:
        raise ValueError(f"E5 unsupported mode values: {invalid_modes}")

    prompt_lengths = core._safe_int_list(
        experiment_config.get("prompt_lengths"),
        key_name="prompt_lengths",
        min_value=8,
    )
    trials_per_length = core._safe_int(experiment_config.get("trials_per_length"), 2)
    if trials_per_length < 1:
        raise ValueError("E5 field 'trials_per_length' must be >= 1.")
    max_new_tokens = core._safe_int(experiment_config.get("max_new_tokens"), 64)
    if max_new_tokens < 1:
        raise ValueError("E5 field 'max_new_tokens' must be >= 1.")
    temperature = core._safe_float(experiment_config.get("temperature"), 0.0)
    top_p = core._safe_float(experiment_config.get("top_p"), 1.0)
    bytes_per_token_estimate = core._safe_int(experiment_config.get("bytes_per_token_estimate"), 2048)

    configured_agg_path = core._safe_str(
        experiment_config.get("aggregated_serving_config"),
        "configs/serving/aggregated.yaml",
    )
    configured_disagg_path = core._safe_str(
        experiment_config.get("disaggregated_serving_config"),
        "configs/serving/disaggregated.yaml",
    )
    agg_serving_path = str(Path(aggregated_serving_config_path or configured_agg_path))
    disagg_serving_path = str(Path(disaggregated_serving_config_path or configured_disagg_path))
    aggregated_serving_config = load_simple_yaml(agg_serving_path)
    disaggregated_serving_config = load_simple_yaml(disagg_serving_path)

    mode_to_serving_config = {
        "aggregated": aggregated_serving_config,
        "disaggregated": disaggregated_serving_config,
    }
    mode_to_allow_simulated: dict[str, bool] = {}
    for mode in modes:
        if allow_simulated_backend is not None:
            mode_to_allow_simulated[mode] = allow_simulated_backend
        else:
            mode_to_allow_simulated[mode] = core._safe_bool(
                mode_to_serving_config[mode].get("allow_simulated"),
                False,
            )
        if strict_backend:
            mode_to_allow_simulated[mode] = False

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": experiment_name,
            "serving_mode": "memory_decomposition",
            "aggregated_serving_config_path": str(Path(agg_serving_path).resolve()),
            "disaggregated_serving_config_path": str(Path(disagg_serving_path).resolve()),
            "seed": seed,
            "strict_backend": strict_backend,
            "prompt_lengths": prompt_lengths,
            "trials_per_length": trials_per_length,
            "modes": modes,
            **core._as_mapping(experiment_config.get("notes", {}), "notes"),
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at = core._utc_now()
    run_wall_start = perf_counter()
    memory_rows: list[dict[str, Any]] = []
    mode_runs: dict[str, Any] = {mode: {"cases": []} for mode in modes}
    all_request_records: list[dict[str, Any]] = []
    for mode in modes:
        serving_config = mode_to_serving_config[mode]
        for prompt_tokens in prompt_lengths:
            case = _run_e5_memory_case(
                mode=mode,
                model_name=model_name,
                serving_config=serving_config,
                experiment_name=experiment_name,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                trials_per_length=trials_per_length,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                warmup=warmup,
                strict_backend=strict_backend,
                allow_simulated_backend=mode_to_allow_simulated[mode],
            )
            mode_runs[mode]["cases"].append(case)
            all_request_records.extend(case["request_records"])

            gpu_run_summary = (
                case.get("profiling", {})
                .get("gpu", {})
                .get("run_summary", {})
            )
            kv_summary = (
                case.get("profiling", {})
                .get("kv_transfer", {})
                .get("summary", {})
            )
            observed_prompt_values = [
                float(record["prompt_tokens_observed"])
                for record in case["request_records"]
                if isinstance(record.get("prompt_tokens_observed"), (int, float))
            ]
            observed_completion_values = [
                float(record["completion_tokens"])
                for record in case["request_records"]
                if isinstance(record.get("completion_tokens"), (int, float))
            ]
            observed_prompt_avg = _safe_average(observed_prompt_values) or float(prompt_tokens)
            observed_completion_avg = _safe_average(observed_completion_values) or float(max_new_tokens)
            estimated_kv_cache_mb = (
                (observed_prompt_avg + observed_completion_avg) * float(bytes_per_token_estimate)
            ) / (1024.0 * 1024.0)
            memory_peak_mb = _to_optional_float(gpu_run_summary.get("memory_used_mb_peak"))
            estimated_non_kv_mb = None
            if memory_peak_mb is not None:
                estimated_non_kv_mb = max(memory_peak_mb - estimated_kv_cache_mb, 0.0)
            memory_rows.append(
                {
                    "mode": mode,
                    "prompt_tokens_target": prompt_tokens,
                    "trials": len(case["request_records"]),
                    "case_success": case["success"],
                    "error": case["error"],
                    "memory_used_mb_avg": _to_optional_float(gpu_run_summary.get("memory_used_mb_avg")),
                    "memory_used_mb_peak": memory_peak_mb,
                    "estimated_kv_cache_mb": estimated_kv_cache_mb,
                    "estimated_non_kv_mb": estimated_non_kv_mb,
                    "kv_transfer_mb": (
                        (_to_optional_float(kv_summary.get("total_transfer_bytes")) or 0.0)
                        / (1024.0 * 1024.0)
                    ),
                    "kv_stall_ratio": _to_optional_float(kv_summary.get("stall_ratio")),
                    "ttft_p95_ms": (
                        case.get("profiling", {})
                        .get("latency", {})
                        .get("summary", {})
                        .get("ttft_ms", {})
                        .get("p95")
                    ),
                    "tpot_p95_ms": (
                        case.get("profiling", {})
                        .get("latency", {})
                        .get("summary", {})
                        .get("tpot_ms", {})
                        .get("p95")
                    ),
                    "backend_modes": case["backend_modes"],
                }
            )
    wall_time_ms = (perf_counter() - run_wall_start) * 1000.0
    finished_at = core._utc_now()

    e5_csv_path = artifacts_dir / "e5_memory_decomposition.csv"
    core._write_csv_rows(
        output_path=e5_csv_path,
        fieldnames=[
            "mode",
            "prompt_tokens_target",
            "trials",
            "case_success",
            "error",
            "memory_used_mb_avg",
            "memory_used_mb_peak",
            "estimated_kv_cache_mb",
            "estimated_non_kv_mb",
            "kv_transfer_mb",
            "kv_stall_ratio",
            "ttft_p95_ms",
            "tpot_p95_ms",
            "backend_modes",
        ],
        rows=memory_rows,
    )

    e5_peak_plot_path = artifacts_dir / "e5_memory_peak_curve.svg"
    series: list[tuple[str, list[float]]] = []
    for mode in modes:
        mode_rows = [row for row in memory_rows if row["mode"] == mode]
        mode_index = {int(row["prompt_tokens_target"]): row for row in mode_rows}
        series.append(
            (
                f"{mode} peak",
                [
                    _to_plot_value((mode_index.get(prompt_tokens, {})).get("memory_used_mb_peak"))
                    for prompt_tokens in prompt_lengths
                ],
            )
        )
    core._write_line_chart_svg(
        output_path=e5_peak_plot_path,
        title=f"{experiment_name}: Peak Memory vs Prompt Length",
        x_label="Prompt Tokens",
        y_label="Peak GPU Memory (MB)",
        x_values=[float(value) for value in prompt_lengths],
        series=series,
    )

    success_cases = [row for row in memory_rows if row["case_success"]]
    success = bool(success_cases)
    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "case_count": len(memory_rows),
        "successful_case_count": len(success_cases),
        "environment_manifest_present": True,
        "csv_generated": e5_csv_path.exists(),
        "plot_generated": e5_peak_plot_path.exists(),
    }
    summary_payload: dict[str, Any] = {
        "run_id": metadata.run_id,
        "experiment_name": experiment_name,
        "success": success,
        "error": None if success_cases else "No successful memory decomposition cases.",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ms": wall_time_ms,
        "mode": "memory_decomposition",
        "backend": "vllm",
        "seed": seed,
        "checks": checks,
        "experiment_config_path": str(Path(experiment_config_path).resolve()),
        "aggregated_serving_config_path": str(Path(agg_serving_path).resolve()),
        "disaggregated_serving_config_path": str(Path(disagg_serving_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "memory_rows": memory_rows,
        "request_records": all_request_records,
        "mode_runs": mode_runs,
        "artifacts": {
            "e5_memory_decomposition_csv": str(e5_csv_path.resolve()),
            "e5_memory_peak_curve_svg": str(e5_peak_plot_path.resolve()),
        },
        "environment_manifest": core._env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }
    summary_path = run_dir / "e5_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return E5RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode="memory_decomposition",
        success=success,
    )


def _build_long_prompt(*, base_prompt: str, target_prompt_tokens: int, filler_token: str) -> str:
    base_tokens = base_prompt.split()
    target = max(target_prompt_tokens, len(base_tokens))
    filler_count = target - len(base_tokens)
    if filler_count <= 0:
        return " ".join(base_tokens)
    filler = " ".join([filler_token] * filler_count)
    return f"{filler} {base_prompt}"


def run_e6(
    *,
    experiment_config_path: str | Path,
    serving_config_path: str | Path = "configs/serving/aggregated.yaml",
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
    benchmark_config_path: str | Path | None = None,
) -> E6RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    serving_config = load_simple_yaml(serving_config_path)
    experiment_name = str(experiment_config.get("name", "e6_yarn_stress")).strip()
    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")
    if "seed" not in experiment_config:
        raise ValueError("E6 requires a fixed integer seed in experiment config.")
    seed = core._safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")

    target_prompt_lengths = core._safe_int_list(
        experiment_config.get("target_prompt_lengths"),
        key_name="target_prompt_lengths",
        min_value=8,
    )
    trials_per_length = core._safe_int(experiment_config.get("trials_per_length"), 1)
    if trials_per_length < 1:
        raise ValueError("E6 field 'trials_per_length' must be >= 1.")
    max_new_tokens = core._safe_int(experiment_config.get("max_new_tokens"), 32)
    if max_new_tokens < 1:
        raise ValueError("E6 field 'max_new_tokens' must be >= 1.")
    temperature = core._safe_float(experiment_config.get("temperature"), 0.0)
    top_p = core._safe_float(experiment_config.get("top_p"), 1.0)
    filler_token = core._safe_str(experiment_config.get("filler_token"), "ctx")

    benchmark_path = str(
        Path(
            benchmark_config_path
            or core._safe_str(experiment_config.get("benchmark_config"), "configs/benchmarks/wp1_smoke.yaml")
        )
    )
    benchmark_suite = load_benchmark_suite(benchmark_path)
    benchmark_samples = [sample for task in benchmark_suite.tasks for sample in task.samples]
    if not benchmark_samples:
        benchmark_samples = [
            BenchmarkSample(
                task="synthetic",
                sample_id="synthetic-001",
                prompt="Summarize the key memory routing behavior in one sentence.",
                reference="memory routing",
            )
        ]

    if allow_simulated_backend is None:
        allow_simulated_backend = core._safe_bool(serving_config.get("allow_simulated"), False)
    if strict_backend:
        allow_simulated_backend = False

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": experiment_name,
            "serving_mode": "aggregated",
            "serving_config_path": str(Path(serving_config_path).resolve()),
            "benchmark_config_path": str(Path(benchmark_path).resolve()),
            "seed": seed,
            "strict_backend": strict_backend,
            "target_prompt_lengths": target_prompt_lengths,
            "trials_per_length": trials_per_length,
            **core._as_mapping(experiment_config.get("notes", {}), "notes"),
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at = core._utc_now()
    wall_start = perf_counter()
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=core._safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode("aggregated")
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    request_records: list[dict[str, Any]] = []
    route_log: list[dict[str, Any]] = []
    backend_modes: dict[str, str] = {}
    success = False
    error_message: str | None = None

    sample_cursor = 0
    for target_prompt_tokens in target_prompt_lengths:
        scheduler: AggregatedScheduler | None = None
        backend: VLLMBackend | None = None
        backend_start_error: str | None = None
        try:
            backend_config, scheduler = core._build_aggregated_components(
                model_name=model_name,
                serving_config=serving_config,
                max_model_len=target_prompt_tokens + max_new_tokens,
                allow_simulated_backend=allow_simulated_backend,
            )
            backend = VLLMBackend(backend_config)
            backend.start()
            backend_modes[f"target_{target_prompt_tokens}"] = backend.mode
            if strict_backend and backend.mode != "vllm":
                raise RuntimeError("Strict backend mode requires real vLLM execution.")
            if warmup:
                backend.warmup()
        except Exception as exc:
            backend_start_error = str(exc)

        if backend_start_error:
            for trial_idx in range(trials_per_length):
                request_records.append(
                    {
                        "target_prompt_tokens": target_prompt_tokens,
                        "trial_index": trial_idx + 1,
                        "success": False,
                        "error": backend_start_error,
                        "task": None,
                        "sample_id": None,
                        "reference": None,
                        "ttft_ms": None,
                        "tpot_ms": None,
                        "prompt_tokens_observed": None,
                        "completion_tokens": None,
                        "quality_score": None,
                    }
                )
            if backend is not None:
                backend.stop()
            continue

        assert scheduler is not None
        assert backend is not None
        for trial_idx in range(trials_per_length):
            sample = benchmark_samples[sample_cursor % len(benchmark_samples)]
            sample_cursor += 1
            prompt = _build_long_prompt(
                base_prompt=sample.prompt,
                target_prompt_tokens=target_prompt_tokens,
                filler_token=filler_token,
            )
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                request_id=(
                    f"{experiment_name}-pt{target_prompt_tokens}-trial{trial_idx + 1:03d}"
                ),
                metadata={
                    "experiment_name": experiment_name,
                    "target_prompt_tokens": target_prompt_tokens,
                    "task": sample.task,
                    "sample_id": sample.sample_id,
                    "seed": seed,
                },
            )
            try:
                _, _, _, generation_result = core._run_single_aggregated_request(
                    backend=backend,
                    scheduler=scheduler,
                    request=request,
                )
                quality = _quality_scores(generation_result.text, sample.reference)
                latency_collector.record_sample(
                    request_id=request.request_id or "e6-request-unknown",
                    ttft_ms=generation_result.ttft_ms,
                    tpot_ms=generation_result.tpot_ms,
                    prompt_tokens=generation_result.prompt_tokens,
                    completion_tokens=generation_result.completion_tokens,
                    metadata={
                        **generation_result.metadata,
                        "target_prompt_tokens": target_prompt_tokens,
                        "task": sample.task,
                        "sample_id": sample.sample_id,
                    },
                )
                request_records.append(
                    {
                        "target_prompt_tokens": target_prompt_tokens,
                        "trial_index": trial_idx + 1,
                        "success": True,
                        "error": None,
                        "task": sample.task,
                        "sample_id": sample.sample_id,
                        "reference": sample.reference,
                        "ttft_ms": generation_result.ttft_ms,
                        "tpot_ms": generation_result.tpot_ms,
                        "prompt_tokens_observed": generation_result.prompt_tokens,
                        "completion_tokens": generation_result.completion_tokens,
                        "quality_score": quality["quality_score"],
                        "token_f1": quality["token_f1"],
                        "contains_reference": quality["contains_reference"],
                        "exact_match": quality["exact_match"],
                    }
                )
            except Exception as exc:
                request_records.append(
                    {
                        "target_prompt_tokens": target_prompt_tokens,
                        "trial_index": trial_idx + 1,
                        "success": False,
                        "error": str(exc),
                        "task": sample.task,
                        "sample_id": sample.sample_id,
                        "reference": sample.reference,
                        "ttft_ms": None,
                        "tpot_ms": None,
                        "prompt_tokens_observed": None,
                        "completion_tokens": None,
                        "quality_score": None,
                    }
                )

        route_log.extend(scheduler.route_log())
        backend.stop()

    latency_collector.stop()
    gpu_collector.stop()
    kv_transfer_collector.stop()

    wall_time_ms = (perf_counter() - wall_start) * 1000.0
    finished_at = core._utc_now()
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)
    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }

    length_summaries: list[dict[str, Any]] = []
    for target_prompt_tokens in target_prompt_lengths:
        entries = [
            record
            for record in request_records
            if core._safe_int(record.get("target_prompt_tokens"), -1) == target_prompt_tokens
        ]
        successes = [record for record in entries if bool(record.get("success"))]
        ttft_values = [float(record["ttft_ms"]) for record in successes if record.get("ttft_ms") is not None]
        tpot_values = [float(record["tpot_ms"]) for record in successes if record.get("tpot_ms") is not None]
        quality_values = [
            float(record["quality_score"])
            for record in successes
            if isinstance(record.get("quality_score"), (int, float))
        ]
        length_summaries.append(
            {
                "target_prompt_tokens": target_prompt_tokens,
                "attempt_count": len(entries),
                "success_count": len(successes),
                "failure_count": len(entries) - len(successes),
                "ttft_ms": core._distribution(ttft_values),
                "tpot_ms": core._distribution(tpot_values),
                "quality_score_avg": _safe_average(quality_values),
                "errors": [
                    record["error"]
                    for record in entries
                    if isinstance(record.get("error"), str) and record["error"].strip()
                ],
            }
        )

    successful_lengths = [
        row["target_prompt_tokens"]
        for row in length_summaries
        if core._safe_int(row.get("success_count"), 0) > 0
    ]
    max_successful_length = max(successful_lengths) if successful_lengths else None
    success = bool(successful_lengths)
    if not success:
        error_message = "No successful YaRN stress runs."

    e6_request_csv_path = artifacts_dir / "e6_context_stress.csv"
    core._write_csv_rows(
        output_path=e6_request_csv_path,
        fieldnames=[
            "target_prompt_tokens",
            "trial_index",
            "success",
            "error",
            "task",
            "sample_id",
            "reference",
            "ttft_ms",
            "tpot_ms",
            "prompt_tokens_observed",
            "completion_tokens",
            "quality_score",
            "token_f1",
            "contains_reference",
            "exact_match",
        ],
        rows=request_records,
    )
    e6_length_csv_path = artifacts_dir / "e6_length_summary.csv"
    core._write_csv_rows(
        output_path=e6_length_csv_path,
        fieldnames=[
            "target_prompt_tokens",
            "attempt_count",
            "success_count",
            "failure_count",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "tpot_p50_ms",
            "tpot_p95_ms",
            "quality_score_avg",
            "errors",
        ],
        rows=[
            {
                "target_prompt_tokens": row["target_prompt_tokens"],
                "attempt_count": row["attempt_count"],
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
                "ttft_p50_ms": (row["ttft_ms"] or {}).get("p50"),
                "ttft_p95_ms": (row["ttft_ms"] or {}).get("p95"),
                "tpot_p50_ms": (row["tpot_ms"] or {}).get("p50"),
                "tpot_p95_ms": (row["tpot_ms"] or {}).get("p95"),
                "quality_score_avg": row["quality_score_avg"],
                "errors": "; ".join(row["errors"]),
            }
            for row in length_summaries
        ],
    )

    plot_x = [float(row["target_prompt_tokens"]) for row in length_summaries]
    e6_ttft_plot_path = artifacts_dir / "e6_ttft_curve.svg"
    core._write_line_chart_svg(
        output_path=e6_ttft_plot_path,
        title=f"{experiment_name}: TTFT vs Context Length",
        x_label="Target Prompt Tokens",
        y_label="TTFT (ms)",
        x_values=plot_x,
        series=[
            ("TTFT p50", [_to_plot_value((row["ttft_ms"] or {}).get("p50")) for row in length_summaries]),
            ("TTFT p95", [_to_plot_value((row["ttft_ms"] or {}).get("p95")) for row in length_summaries]),
        ],
    )
    e6_tpot_plot_path = artifacts_dir / "e6_tpot_curve.svg"
    core._write_line_chart_svg(
        output_path=e6_tpot_plot_path,
        title=f"{experiment_name}: TPOT vs Context Length",
        x_label="Target Prompt Tokens",
        y_label="TPOT (ms/token)",
        x_values=plot_x,
        series=[
            ("TPOT p50", [_to_plot_value((row["tpot_ms"] or {}).get("p50")) for row in length_summaries]),
            ("TPOT p95", [_to_plot_value((row["tpot_ms"] or {}).get("p95")) for row in length_summaries]),
        ],
    )

    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "attempted_lengths": len(target_prompt_lengths),
        "successful_lengths": len(successful_lengths),
        "max_successful_prompt_tokens": max_successful_length,
        "environment_manifest_present": True,
    }
    summary_payload: dict[str, Any] = {
        "run_id": metadata.run_id,
        "experiment_name": experiment_name,
        "success": success,
        "error": error_message,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ms": wall_time_ms,
        "mode": "aggregated",
        "backend": "vllm",
        "backend_modes": backend_modes,
        "seed": seed,
        "checks": checks,
        "experiment_config_path": str(Path(experiment_config_path).resolve()),
        "serving_config_path": str(Path(serving_config_path).resolve()),
        "benchmark_config_path": str(Path(benchmark_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "profiling": profiling_snapshot,
        "route_log": route_log,
        "request_records": request_records,
        "length_summaries": length_summaries,
        "max_successful_prompt_tokens": max_successful_length,
        "benchmark_suite": {
            "name": benchmark_suite.name,
            "config_path": benchmark_suite.config_path,
            "tasks": [
                {
                    "name": task.name,
                    "source": task.source,
                    "sample_count": len(task.samples),
                    "sample_ids": [sample.sample_id for sample in task.samples],
                }
                for task in benchmark_suite.tasks
            ],
        },
        "artifacts": {
            "e6_context_stress_csv": str(e6_request_csv_path.resolve()),
            "e6_length_summary_csv": str(e6_length_csv_path.resolve()),
            "e6_ttft_curve_svg": str(e6_ttft_plot_path.resolve()),
            "e6_tpot_curve_svg": str(e6_tpot_plot_path.resolve()),
        },
        "environment_manifest": core._env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }
    summary_path = run_dir / "e6_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    return E6RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode="aggregated",
        success=success,
    )
