from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

from context_research.config.io import load_simple_yaml
from context_research.config.schema import create_run_metadata, run_metadata_to_dict, write_run_metadata
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
class E0RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


@dataclass(frozen=True)
class E1RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


@dataclass(frozen=True)
class E2RunResult:
    run_id: str
    run_dir: str
    metadata_path: str
    summary_path: str
    mode: str
    success: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_prompt(max_prompt_tokens: int) -> str:
    target_tokens = max(8, min(max_prompt_tokens, 64))
    return " ".join(f"tok{i}" for i in range(target_tokens))


def _make_prompt_exact(prompt_tokens: int) -> str:
    target_tokens = max(1, prompt_tokens)
    return " ".join(f"tok{i}" for i in range(target_tokens))


def _safe_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit())):
            return int(stripped)
    if isinstance(value, float):
        return int(value)
    return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _safe_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _safe_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _as_mapping(value: Any, key_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key_name}' to be a mapping.")
    return value


def _safe_int_list(value: Any, *, key_name: str, min_value: int = 0) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"Expected '{key_name}' to be a list of integers.")
    parsed: list[int] = []
    for item in value:
        integer = _safe_int(item, min_value - 1)
        if integer < min_value:
            raise ValueError(
                f"Expected every '{key_name}' entry to be >= {min_value}, got {item!r}."
            )
        parsed.append(integer)
    if not parsed:
        raise ValueError(f"Expected '{key_name}' to have at least one value.")
    return parsed


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if quantile <= 0:
        return min(values)
    if quantile >= 100:
        return max(values)

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (quantile / 100.0)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = position - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return lower + ((upper - lower) * weight)


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "avg": None,
            "p50": None,
            "p95": None,
        }
    avg = sum(values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "avg": avg,
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
    }


def _write_csv_rows(
    *,
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_svg_polyline_points(
    *,
    xs: list[float],
    ys: list[float],
    chart_x0: float,
    chart_y0: float,
    chart_w: float,
    chart_h: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    points: list[str] = []
    x_range = x_max - x_min
    y_range = y_max - y_min
    for x_value, y_value in zip(xs, ys):
        normalized_x = 0.5 if x_range == 0 else (x_value - x_min) / x_range
        normalized_y = 0.5 if y_range == 0 else (y_value - y_min) / y_range
        svg_x = chart_x0 + (normalized_x * chart_w)
        svg_y = chart_y0 + chart_h - (normalized_y * chart_h)
        points.append(f"{svg_x:.2f},{svg_y:.2f}")
    return " ".join(points)


def _write_line_chart_svg(
    *,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
) -> None:
    width = 980
    height = 520
    margin_left = 86
    margin_right = 24
    margin_top = 54
    margin_bottom = 92
    chart_x0 = float(margin_left)
    chart_y0 = float(margin_top)
    chart_w = float(width - margin_left - margin_right)
    chart_h = float(height - margin_top - margin_bottom)

    filtered_series: list[tuple[str, list[float], list[float]]] = []
    for label, y_values in series:
        points = [(float(x), float(y)) for x, y in zip(x_values, y_values)]
        filtered = [(x, y) for x, y in points if x == x and y == y]
        if not filtered:
            continue
        filtered_series.append(
            (
                label,
                [point[0] for point in filtered],
                [point[1] for point in filtered],
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not filtered_series:
        output_path.write_text(
            (
                f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
                "<rect x='0' y='0' width='100%' height='100%' fill='white'/>"
                f"<text x='{width / 2:.0f}' y='{height / 2:.0f}' text-anchor='middle' "
                "font-family='monospace' font-size='16' fill='#111827'>No data</text>"
                "</svg>"
            ),
            encoding="utf-8",
        )
        return

    all_x: list[float] = []
    all_y: list[float] = []
    for _, xs, ys in filtered_series:
        all_x.extend(xs)
        all_y.extend(ys)

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c"]
    ticks = 5
    svg_lines: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        f"<text x='{width / 2:.2f}' y='28' text-anchor='middle' font-family='monospace' "
        f"font-size='18' fill='#111827'>{escape(title)}</text>",
        f"<line x1='{chart_x0:.2f}' y1='{chart_y0 + chart_h:.2f}' x2='{chart_x0 + chart_w:.2f}' "
        f"y2='{chart_y0 + chart_h:.2f}' stroke='#374151' stroke-width='1.5'/>",
        f"<line x1='{chart_x0:.2f}' y1='{chart_y0:.2f}' x2='{chart_x0:.2f}' y2='{chart_y0 + chart_h:.2f}' "
        "stroke='#374151' stroke-width='1.5'/>",
    ]

    for tick in range(ticks + 1):
        tick_ratio = tick / ticks
        x_value = x_min + ((x_max - x_min) * tick_ratio)
        y_value = y_min + ((y_max - y_min) * (1.0 - tick_ratio))
        tick_x = chart_x0 + (chart_w * tick_ratio)
        tick_y = chart_y0 + (chart_h * tick_ratio)
        svg_lines.extend(
            [
                f"<line x1='{tick_x:.2f}' y1='{chart_y0 + chart_h:.2f}' x2='{tick_x:.2f}' "
                f"y2='{chart_y0 + chart_h + 6:.2f}' stroke='#374151' stroke-width='1'/>",
                f"<text x='{tick_x:.2f}' y='{chart_y0 + chart_h + 24:.2f}' text-anchor='middle' "
                f"font-family='monospace' font-size='12' fill='#374151'>{x_value:.0f}</text>",
                f"<line x1='{chart_x0 - 6:.2f}' y1='{tick_y:.2f}' x2='{chart_x0:.2f}' y2='{tick_y:.2f}' "
                "stroke='#374151' stroke-width='1'/>",
                f"<text x='{chart_x0 - 10:.2f}' y='{tick_y + 4:.2f}' text-anchor='end' "
                f"font-family='monospace' font-size='12' fill='#374151'>{y_value:.2f}</text>",
                f"<line x1='{chart_x0:.2f}' y1='{tick_y:.2f}' x2='{chart_x0 + chart_w:.2f}' y2='{tick_y:.2f}' "
                "stroke='#e5e7eb' stroke-width='1'/>",
            ]
        )

    for index, (label, xs, ys) in enumerate(filtered_series):
        color = colors[index % len(colors)]
        polyline_points = _to_svg_polyline_points(
            xs=xs,
            ys=ys,
            chart_x0=chart_x0,
            chart_y0=chart_y0,
            chart_w=chart_w,
            chart_h=chart_h,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        svg_lines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2.2' points='{polyline_points}'/>"
        )
        for point_x, point_y in zip(xs, ys):
            circle_points = _to_svg_polyline_points(
                xs=[point_x],
                ys=[point_y],
                chart_x0=chart_x0,
                chart_y0=chart_y0,
                chart_w=chart_w,
                chart_h=chart_h,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            point_svg_x, point_svg_y = circle_points.split(",")
            svg_lines.append(
                f"<circle cx='{point_svg_x}' cy='{point_svg_y}' r='3.2' fill='{color}'/>"
            )
        legend_x = chart_x0 + 18 + (index * 210)
        legend_y = chart_y0 + chart_h + 48
        svg_lines.extend(
            [
                f"<line x1='{legend_x:.2f}' y1='{legend_y:.2f}' x2='{legend_x + 28:.2f}' y2='{legend_y:.2f}' "
                f"stroke='{color}' stroke-width='3'/>",
                f"<text x='{legend_x + 36:.2f}' y='{legend_y + 4:.2f}' font-family='monospace' "
                f"font-size='12' fill='#111827'>{escape(label)}</text>",
            ]
        )

    svg_lines.extend(
        [
            f"<text x='{chart_x0 + (chart_w / 2):.2f}' y='{height - 24:.2f}' text-anchor='middle' "
            f"font-family='monospace' font-size='13' fill='#111827'>{escape(x_label)}</text>",
            f"<text x='22' y='{chart_y0 + (chart_h / 2):.2f}' text-anchor='middle' "
            f"font-family='monospace' font-size='13' fill='#111827' "
            f"transform='rotate(-90 22,{chart_y0 + (chart_h / 2):.2f})'>{escape(y_label)}</text>",
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def _collect_nvidia_manifest() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False, "gpus": []}

    gpus: list[dict[str, Any]] = []
    for raw_line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) != 4:
            continue
        index = _safe_int(parts[0], -1)
        memory_mb = _safe_int(parts[2], -1)
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mb": memory_mb if memory_mb >= 0 else None,
                "driver_version": parts[3] or None,
            }
        )
    return {"available": True, "gpus": gpus}


def _discover_gpu_indices() -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    indices: list[int] = []
    for raw_line in result.stdout.strip().splitlines():
        parsed = _safe_int(raw_line.strip(), -1)
        if parsed >= 0:
            indices.append(parsed)
    return sorted(indices)


def _parse_visible_devices(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
        return None

    if isinstance(value, list):
        parsed: list[str] = []
        for item in value:
            index = _safe_int(item, -1)
            if index >= 0:
                parsed.append(str(index))
        if parsed:
            return ",".join(parsed)
    return None


def _visible_devices_from_indices(indices: list[int]) -> str | None:
    if not indices:
        return None
    return ",".join(str(index) for index in indices)


def _resolve_disaggregated_gpu_assignment(serving_config: dict[str, Any]) -> dict[str, Any]:
    vllm_config = _as_mapping(serving_config.get("vllm"), "vllm")
    tensor_parallel_size = max(1, _safe_int(vllm_config.get("tensor_parallel_size"), 1))
    available_indices = _discover_gpu_indices()

    prefill_override = _parse_visible_devices(serving_config.get("prefill_visible_devices"))
    decode_override = _parse_visible_devices(serving_config.get("decode_visible_devices"))

    assignment: dict[str, Any] = {
        "strategy": "auto",
        "tensor_parallel_size": tensor_parallel_size,
        "available_gpu_indices": available_indices,
        "prefill_visible_devices": prefill_override,
        "decode_visible_devices": decode_override,
        "warning": None,
    }

    if prefill_override or decode_override:
        assignment["strategy"] = "manual_override"
        if not prefill_override or not decode_override:
            assignment["warning"] = (
                "Manual GPU override was partial; one backend has no explicit visible-device assignment."
            )
        return assignment

    if len(available_indices) < 2:
        assignment["strategy"] = "single_or_unknown_gpu"
        assignment["warning"] = (
            "Auto GPU assignment requires at least 2 visible GPUs. "
            "Disaggregated backends will use default placement."
        )
        return assignment

    required_for_isolation = tensor_parallel_size * 2
    if len(available_indices) >= required_for_isolation:
        prefill_indices = available_indices[:tensor_parallel_size]
        decode_indices = available_indices[tensor_parallel_size:required_for_isolation]
        assignment["strategy"] = "auto_non_overlapping"
        assignment["prefill_visible_devices"] = _visible_devices_from_indices(prefill_indices)
        assignment["decode_visible_devices"] = _visible_devices_from_indices(decode_indices)
        return assignment

    if len(available_indices) >= tensor_parallel_size:
        shared_indices = available_indices[:tensor_parallel_size]
        shared_visible = _visible_devices_from_indices(shared_indices)
        assignment["strategy"] = "auto_shared_due_to_insufficient_gpus"
        assignment["prefill_visible_devices"] = shared_visible
        assignment["decode_visible_devices"] = shared_visible
        assignment["warning"] = (
            f"Only {len(available_indices)} GPUs visible for tensor_parallel_size={tensor_parallel_size}. "
            "Prefill/decode will share GPUs."
        )
        return assignment

    assignment["strategy"] = "insufficient_for_tensor_parallel"
    assignment["warning"] = (
        f"Visible GPUs ({len(available_indices)}) are below tensor_parallel_size "
        f"({tensor_parallel_size}). Falling back to default placement."
    )
    return assignment


def _env_manifest() -> dict[str, Any]:
    lightning_keys = [
        "LIGHTNING_STUDIO_URL",
        "LIGHTNING_RUN_URL",
        "LIGHTNING_WORKSPACE_URL",
    ]
    lightning_env = {key: value for key in lightning_keys if (value := os.environ.get(key))}
    return {
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "python_executable": os.environ.get("PYTHON_EXECUTABLE") or os.sys.executable,
        "python_version": platform.python_version(),
        "hf_home": os.environ.get("HF_HOME"),
        "lightning_env": lightning_env,
        "nvidia": _collect_nvidia_manifest(),
    }


def _build_vllm_backend_config(
    *,
    model_name: str,
    serving_config: dict[str, Any],
    max_model_len: int,
    allow_simulated_backend: bool,
    visible_devices: str | None = None,
    device: str | None = None,
) -> VLLMBackendConfig:
    vllm_config = _as_mapping(serving_config.get("vllm"), "vllm")
    return VLLMBackendConfig(
        model=model_name,
        tensor_parallel_size=_safe_int(vllm_config.get("tensor_parallel_size"), 1),
        gpu_memory_utilization=_safe_float(vllm_config.get("gpu_memory_utilization"), 0.9),
        max_model_len=max_model_len,
        trust_remote_code=_safe_bool(vllm_config.get("trust_remote_code"), True),
        dtype=_safe_optional_str(vllm_config.get("dtype")),
        enforce_eager=_safe_bool(vllm_config.get("enforce_eager"), True),
        enable_thinking=_safe_bool(vllm_config.get("enable_thinking"), False),
        simulate_if_unavailable=allow_simulated_backend,
        visible_devices=visible_devices,
        device=device,
    )


def run_e0(
    *,
    experiment_config_path: str | Path,
    serving_config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    request_prompt: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
) -> E0RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    serving_config = load_simple_yaml(serving_config_path)

    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")

    if "seed" not in experiment_config:
        raise ValueError("E0 requires a fixed integer seed in experiment config.")
    seed = _safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")

    mode = str(serving_config.get("mode", "aggregated")).strip().lower()
    backend_name = str(serving_config.get("backend", "vllm")).strip().lower()
    if backend_name != "vllm":
        raise ValueError(f"Unsupported backend '{backend_name}'. Expected 'vllm'.")
    if mode not in {"aggregated", "disaggregated"}:
        raise ValueError(
            f"Unsupported serving mode '{mode}'. Expected 'aggregated' or 'disaggregated'."
        )

    if allow_simulated_backend is None:
        allow_simulated_backend = _safe_bool(serving_config.get("allow_simulated"), False)
    if strict_backend:
        allow_simulated_backend = False

    exp_notes_raw = experiment_config.get("notes", {})
    exp_notes = _as_mapping(exp_notes_raw, "notes")

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": str(experiment_config.get("name", "e0_smoke")),
            "serving_config_path": str(Path(serving_config_path).resolve()),
            "serving_mode": mode,
            "seed": seed,
            "strict_backend": strict_backend,
            **exp_notes,
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent

    max_prompt_tokens = _safe_int(experiment_config.get("max_prompt_tokens"), 1024)
    max_new_tokens = _safe_int(experiment_config.get("max_new_tokens"), 64)
    temperature = _safe_float(experiment_config.get("temperature"), 0.0)
    top_p = _safe_float(experiment_config.get("top_p"), 1.0)

    prompt = request_prompt or _make_prompt(max_prompt_tokens=max_prompt_tokens)
    request_id = "e0-smoke-req-0001"
    request = GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        request_id=request_id,
        metadata={
            "experiment_name": str(experiment_config.get("name", "e0_smoke")),
            "max_prompt_tokens": max_prompt_tokens,
            "seed": seed,
        },
    )

    gpu_assignment: dict[str, Any] = {
        "strategy": "not_applicable",
        "warning": None,
        "tensor_parallel_size": None,
        "available_gpu_indices": [],
        "prefill_visible_devices": None,
        "decode_visible_devices": None,
    }
    aggregated_backend_config: VLLMBackendConfig | None = None
    prefill_backend_config: VLLMBackendConfig | None = None
    decode_backend_config: VLLMBackendConfig | None = None
    if mode == "aggregated":
        aggregated_backend_config = _build_vllm_backend_config(
            model_name=model_name,
            serving_config=serving_config,
            max_model_len=max_prompt_tokens + max_new_tokens,
            allow_simulated_backend=allow_simulated_backend,
            visible_devices=_parse_visible_devices(serving_config.get("visible_devices")),
            device=_safe_optional_str(serving_config.get("device")),
        )
        scheduler: AggregatedScheduler | DisaggregatedScheduler = AggregatedScheduler(
            pool_name=_safe_str(serving_config.get("pool_name"), "aggregated"),
            pool_size=_safe_int(serving_config.get("pool_size"), 1),
        )
    else:
        gpu_assignment = _resolve_disaggregated_gpu_assignment(serving_config)
        prefill_backend_config = _build_vllm_backend_config(
            model_name=model_name,
            serving_config=serving_config,
            max_model_len=max_prompt_tokens + max_new_tokens,
            allow_simulated_backend=allow_simulated_backend,
            visible_devices=_safe_optional_str(gpu_assignment.get("prefill_visible_devices")),
            device=_safe_optional_str(serving_config.get("prefill_device")),
        )
        decode_backend_config = _build_vllm_backend_config(
            model_name=model_name,
            serving_config=serving_config,
            max_model_len=max_prompt_tokens + max_new_tokens,
            allow_simulated_backend=allow_simulated_backend,
            visible_devices=_safe_optional_str(gpu_assignment.get("decode_visible_devices")),
            device=_safe_optional_str(serving_config.get("decode_device")),
        )
        broker = LocalQueuePDBroker(
            bytes_per_token=_safe_int(serving_config.get("bytes_per_token"), 2048)
        )
        scheduler = DisaggregatedScheduler(
            prefill_pool=_safe_str(serving_config.get("prefill_pool_name"), "prefill"),
            prefill_pool_size=_safe_int(serving_config.get("prefill_pool_size"), 1),
            decode_pool=_safe_str(serving_config.get("decode_pool_name"), "decode"),
            decode_pool_size=_safe_int(serving_config.get("decode_pool_size"), 1),
            broker=broker,
        )

    started_at = _utc_now()
    wall_start = perf_counter()
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=_safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode(mode)
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    generation_result = None
    prefill_result: dict[str, Any] | None = None
    prefill_pool = None
    decode_pool = None
    broker_snapshot: dict[str, Any] | None = None
    broker_completed_handoffs: list[dict[str, Any]] = []
    success = False
    error_message = None
    backend_modes: dict[str, str] = {}

    aggregated_backend: VLLMBackend | None = None
    prefill_backend: VLLMBackend | None = None
    decode_backend: VLLMBackend | None = None

    try:
        if mode == "aggregated":
            if aggregated_backend_config is None:
                raise RuntimeError("Missing aggregated backend config.")
            aggregated_backend = VLLMBackend(aggregated_backend_config)
            aggregated_backend.start()
            if warmup:
                aggregated_backend.warmup()
            prefill_pool = scheduler.route_prefill(request)
            prefill_result = aggregated_backend.prefill(request)
            decode_pool = scheduler.route_decode(request.request_id or "")
            generation_result = aggregated_backend.generate(request)
            latency_collector.record_sample(
                request_id=request.request_id or "e0-smoke-req-unknown",
                ttft_ms=generation_result.ttft_ms,
                tpot_ms=generation_result.tpot_ms,
                prompt_tokens=generation_result.prompt_tokens,
                completion_tokens=generation_result.completion_tokens,
                metadata=generation_result.metadata,
            )
            backend_modes["aggregated"] = aggregated_backend.mode
        else:
            if prefill_backend_config is None or decode_backend_config is None:
                raise RuntimeError("Missing disaggregated backend configs.")
            prefill_backend = VLLMBackend(prefill_backend_config)
            decode_backend = VLLMBackend(decode_backend_config)
            prefill_backend.start()
            decode_backend.start()
            if warmup:
                prefill_backend.warmup()
                decode_backend.warmup()
            prefill_pool = scheduler.route_prefill(request)
            prefill_result = prefill_backend.prefill(request)
            decode_pool = scheduler.route_decode(request.request_id or "")
            generation_result = decode_backend.generate(request)
            latency_collector.record_sample(
                request_id=request.request_id or "e0-smoke-req-unknown",
                ttft_ms=generation_result.ttft_ms,
                tpot_ms=generation_result.tpot_ms,
                prompt_tokens=generation_result.prompt_tokens,
                completion_tokens=generation_result.completion_tokens,
                metadata=generation_result.metadata,
            )
            backend_modes["prefill"] = prefill_backend.mode
            backend_modes["decode"] = decode_backend.mode

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
    finished_at = _utc_now()

    route_log = scheduler.route_log()
    if isinstance(scheduler, DisaggregatedScheduler):
        broker_snapshot = scheduler.broker.snapshot()
        broker_completed_handoffs = scheduler.broker.completed_handoffs()
        kv_transfer_collector.ingest_completed_handoffs(broker_completed_handoffs)
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)

    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }

    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "all_backends_real_vllm": bool(backend_modes)
        and all(mode_name == "vllm" for mode_name in backend_modes.values()),
        "generated_output": generation_result is not None,
        "environment_manifest_present": True,
    }

    summary_payload: dict[str, Any] = {
        "run_id": metadata.run_id,
        "experiment_name": str(experiment_config.get("name", "e0_smoke")),
        "success": success,
        "error": error_message,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_ms": wall_time_ms,
        "mode": mode,
        "backend": backend_name,
        "backend_modes": backend_modes,
        "seed": seed,
        "checks": checks,
        "experiment_config_path": str(Path(experiment_config_path).resolve()),
        "serving_config_path": str(Path(serving_config_path).resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "request": asdict(request),
        "prefill_result": prefill_result,
        "generation_result": asdict(generation_result) if generation_result else None,
        "route_log": route_log,
        "prefill_pool": prefill_pool,
        "decode_pool": decode_pool,
        "gpu_assignment": gpu_assignment,
        "broker_snapshot": broker_snapshot,
        "broker_completed_handoffs": broker_completed_handoffs,
        "profiling": profiling_snapshot,
        "environment_manifest": _env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }

    summary_path = run_dir / "e0_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return E0RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode=mode,
        success=success,
    )


def _build_generation_request(
    *,
    request_id: str,
    prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    experiment_name: str,
) -> GenerationRequest:
    prompt = _make_prompt_exact(prompt_tokens=prompt_tokens)
    return GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        request_id=request_id,
        metadata={
            "experiment_name": experiment_name,
            "max_prompt_tokens": prompt_tokens,
            "seed": seed,
        },
    )


def _build_aggregated_components(
    *,
    model_name: str,
    serving_config: dict[str, Any],
    max_model_len: int,
    allow_simulated_backend: bool,
) -> tuple[VLLMBackendConfig, AggregatedScheduler]:
    mode = str(serving_config.get("mode", "aggregated")).strip().lower()
    backend_name = str(serving_config.get("backend", "vllm")).strip().lower()
    if backend_name != "vllm":
        raise ValueError(f"Unsupported backend '{backend_name}'. Expected 'vllm'.")
    if mode != "aggregated":
        raise ValueError(
            f"E1/E2 currently support aggregated mode only; got serving mode '{mode}'."
        )

    backend_config = _build_vllm_backend_config(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=max_model_len,
        allow_simulated_backend=allow_simulated_backend,
        visible_devices=_parse_visible_devices(serving_config.get("visible_devices")),
        device=_safe_optional_str(serving_config.get("device")),
    )
    scheduler = AggregatedScheduler(
        pool_name=_safe_str(serving_config.get("pool_name"), "aggregated"),
        pool_size=_safe_int(serving_config.get("pool_size"), 1),
    )
    return (backend_config, scheduler)


def _run_single_aggregated_request(
    *,
    backend: VLLMBackend,
    scheduler: AggregatedScheduler,
    request: GenerationRequest,
) -> tuple[str, str, dict[str, Any], Any]:
    prefill_pool = scheduler.route_prefill(request)
    prefill_result = backend.prefill(request)
    decode_pool = scheduler.route_decode(request.request_id or "")
    generation_result = backend.generate(request)
    return (prefill_pool, decode_pool, prefill_result, generation_result)


def run_e1(
    *,
    experiment_config_path: str | Path,
    serving_config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
) -> E1RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    serving_config = load_simple_yaml(serving_config_path)
    experiment_name = str(experiment_config.get("name", "e1_aggregated_latency"))

    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")
    if "seed" not in experiment_config:
        raise ValueError("E1 requires a fixed integer seed in experiment config.")
    seed = _safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")

    prompt_lengths = _safe_int_list(
        experiment_config.get("prompt_lengths"),
        key_name="prompt_lengths",
        min_value=8,
    )
    trials_per_length = _safe_int(experiment_config.get("trials_per_length"), 3)
    if trials_per_length < 1:
        raise ValueError("E1 field 'trials_per_length' must be >= 1.")
    max_new_tokens = _safe_int(experiment_config.get("max_new_tokens"), 64)
    if max_new_tokens < 1:
        raise ValueError("E1 field 'max_new_tokens' must be >= 1.")
    temperature = _safe_float(experiment_config.get("temperature"), 0.0)
    top_p = _safe_float(experiment_config.get("top_p"), 1.0)
    if allow_simulated_backend is None:
        allow_simulated_backend = _safe_bool(serving_config.get("allow_simulated"), False)
    if strict_backend:
        allow_simulated_backend = False

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": experiment_name,
            "serving_config_path": str(Path(serving_config_path).resolve()),
            "serving_mode": "aggregated",
            "seed": seed,
            "strict_backend": strict_backend,
            "prompt_lengths": prompt_lengths,
            "trials_per_length": trials_per_length,
            **_as_mapping(experiment_config.get("notes", {}), "notes"),
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    backend_config, scheduler = _build_aggregated_components(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=max(prompt_lengths) + max_new_tokens,
        allow_simulated_backend=allow_simulated_backend,
    )

    started_at = _utc_now()
    wall_start = perf_counter()
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=_safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode("aggregated")
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    backend_modes: dict[str, str] = {}
    trial_records: list[dict[str, Any]] = []
    success = False
    error_message: str | None = None
    backend: VLLMBackend | None = None
    try:
        backend = VLLMBackend(backend_config)
        backend.start()
        backend_modes["aggregated"] = backend.mode
        if warmup:
            backend.warmup()

        for prompt_tokens in prompt_lengths:
            for trial_idx in range(trials_per_length):
                request_id = f"{experiment_name}-pt{prompt_tokens}-trial{trial_idx + 1:03d}"
                request = _build_generation_request(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    experiment_name=experiment_name,
                )
                request_started = perf_counter()
                prefill_pool, decode_pool, prefill_result, generation_result = _run_single_aggregated_request(
                    backend=backend,
                    scheduler=scheduler,
                    request=request,
                )
                request_wall_ms = (perf_counter() - request_started) * 1000.0

                latency_collector.record_sample(
                    request_id=request_id,
                    ttft_ms=generation_result.ttft_ms,
                    tpot_ms=generation_result.tpot_ms,
                    prompt_tokens=generation_result.prompt_tokens,
                    completion_tokens=generation_result.completion_tokens,
                    metadata=generation_result.metadata,
                )
                trial_records.append(
                    {
                        "request_id": request_id,
                        "prompt_tokens_target": prompt_tokens,
                        "trial_index": trial_idx + 1,
                        "prompt_tokens_observed": generation_result.prompt_tokens,
                        "completion_tokens": generation_result.completion_tokens,
                        "ttft_ms": generation_result.ttft_ms,
                        "tpot_ms": generation_result.tpot_ms,
                        "prefill_ms": _safe_float(prefill_result.get("prefill_ms"), 0.0),
                        "request_wall_ms": request_wall_ms,
                        "prefill_pool": prefill_pool,
                        "decode_pool": decode_pool,
                        "latency_source": _safe_optional_str(
                            _as_mapping(generation_result.metadata, "generation_result.metadata").get(
                                "latency_source"
                            )
                        ),
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
    finished_at = _utc_now()
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)
    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }

    prompt_summaries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for prompt_tokens in prompt_lengths:
        prompt_trials = [record for record in trial_records if record["prompt_tokens_target"] == prompt_tokens]
        ttft_values = [float(record["ttft_ms"]) for record in prompt_trials]
        tpot_values = [float(record["tpot_ms"]) for record in prompt_trials]
        prefill_values = [float(record["prefill_ms"]) for record in prompt_trials]
        summary = {
            "prompt_tokens": prompt_tokens,
            "trial_count": len(prompt_trials),
            "ttft_ms": _distribution(ttft_values),
            "tpot_ms": _distribution(tpot_values),
            "prefill_ms": _distribution(prefill_values),
        }
        prompt_summaries.append(summary)
        csv_rows.append(
            {
                "prompt_tokens": prompt_tokens,
                "trial_count": len(prompt_trials),
                "ttft_avg_ms": summary["ttft_ms"]["avg"],
                "ttft_p50_ms": summary["ttft_ms"]["p50"],
                "ttft_p95_ms": summary["ttft_ms"]["p95"],
                "tpot_avg_ms": summary["tpot_ms"]["avg"],
                "tpot_p50_ms": summary["tpot_ms"]["p50"],
                "tpot_p95_ms": summary["tpot_ms"]["p95"],
                "prefill_avg_ms": summary["prefill_ms"]["avg"],
            }
        )

    e1_csv_path = artifacts_dir / "e1_latency_curve.csv"
    _write_csv_rows(
        output_path=e1_csv_path,
        fieldnames=[
            "prompt_tokens",
            "trial_count",
            "ttft_avg_ms",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "tpot_avg_ms",
            "tpot_p50_ms",
            "tpot_p95_ms",
            "prefill_avg_ms",
        ],
        rows=csv_rows,
    )

    plot_x = [float(item["prompt_tokens"]) for item in prompt_summaries]
    e1_ttft_plot_path = artifacts_dir / "e1_ttft_curve.svg"
    _write_line_chart_svg(
        output_path=e1_ttft_plot_path,
        title=f"{experiment_name}: TTFT vs Prompt Length",
        x_label="Prompt Tokens",
        y_label="TTFT (ms)",
        x_values=plot_x,
        series=[
            ("TTFT p50", [float(item["ttft_ms"]["p50"] or 0.0) for item in prompt_summaries]),
            ("TTFT p95", [float(item["ttft_ms"]["p95"] or 0.0) for item in prompt_summaries]),
        ],
    )
    e1_tpot_plot_path = artifacts_dir / "e1_tpot_curve.svg"
    _write_line_chart_svg(
        output_path=e1_tpot_plot_path,
        title=f"{experiment_name}: TPOT vs Prompt Length",
        x_label="Prompt Tokens",
        y_label="TPOT (ms/token)",
        x_values=plot_x,
        series=[
            ("TPOT p50", [float(item["tpot_ms"]["p50"] or 0.0) for item in prompt_summaries]),
            ("TPOT p95", [float(item["tpot_ms"]["p95"] or 0.0) for item in prompt_summaries]),
        ],
    )

    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "all_backends_real_vllm": bool(backend_modes)
        and all(mode_name == "vllm" for mode_name in backend_modes.values()),
        "executed_trials": len(trial_records),
        "plots_generated": e1_ttft_plot_path.exists() and e1_tpot_plot_path.exists(),
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
        "metadata_path": str(metadata_path.resolve()),
        "profiling": profiling_snapshot,
        "route_log": scheduler.route_log(),
        "trial_records": trial_records,
        "latency_by_prompt_tokens": prompt_summaries,
        "artifacts": {
            "e1_latency_curve_csv": str(e1_csv_path.resolve()),
            "e1_ttft_curve_svg": str(e1_ttft_plot_path.resolve()),
            "e1_tpot_curve_svg": str(e1_tpot_plot_path.resolve()),
        },
        "environment_manifest": _env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }

    summary_path = run_dir / "e1_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return E1RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode="aggregated",
        success=success,
    )


def run_e2(
    *,
    experiment_config_path: str | Path,
    serving_config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    warmup: bool = True,
    strict_backend: bool = True,
    allow_simulated_backend: bool | None = None,
) -> E2RunResult:
    experiment_config = load_simple_yaml(experiment_config_path)
    serving_config = load_simple_yaml(serving_config_path)
    experiment_name = str(experiment_config.get("name", "e2_batch_concurrency"))

    model_name = str(experiment_config.get("model", "")).strip()
    if not model_name:
        raise ValueError("Experiment config must include non-empty 'model'.")
    if "seed" not in experiment_config:
        raise ValueError("E2 requires a fixed integer seed in experiment config.")
    seed = _safe_int(experiment_config.get("seed"), -1)
    if seed < 0:
        raise ValueError("Experiment config field 'seed' must be a non-negative integer.")

    concurrency_levels = _safe_int_list(
        experiment_config.get("concurrency_levels"),
        key_name="concurrency_levels",
        min_value=1,
    )
    waves_per_level = _safe_int(experiment_config.get("waves_per_level"), 3)
    if waves_per_level < 1:
        raise ValueError("E2 field 'waves_per_level' must be >= 1.")
    prompt_tokens = _safe_int(experiment_config.get("prompt_tokens"), 4096)
    max_new_tokens = _safe_int(experiment_config.get("max_new_tokens"), 64)
    if prompt_tokens < 8:
        raise ValueError("E2 field 'prompt_tokens' must be >= 8.")
    if max_new_tokens < 1:
        raise ValueError("E2 field 'max_new_tokens' must be >= 1.")
    temperature = _safe_float(experiment_config.get("temperature"), 0.0)
    top_p = _safe_float(experiment_config.get("top_p"), 1.0)
    ttft_slo_ms = _safe_float(experiment_config.get("ttft_slo_ms"), 4000.0)
    tpot_slo_ms = _safe_float(experiment_config.get("tpot_slo_ms"), 120.0)
    if allow_simulated_backend is None:
        allow_simulated_backend = _safe_bool(serving_config.get("allow_simulated"), False)
    if strict_backend:
        allow_simulated_backend = False

    metadata = create_run_metadata(
        model_name=model_name,
        config_path=experiment_config_path,
        run_id=run_id,
        notes={
            "experiment": experiment_name,
            "serving_config_path": str(Path(serving_config_path).resolve()),
            "serving_mode": "aggregated",
            "seed": seed,
            "strict_backend": strict_backend,
            "concurrency_levels": concurrency_levels,
            "waves_per_level": waves_per_level,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
            **_as_mapping(experiment_config.get("notes", {}), "notes"),
        },
    )
    metadata_path = write_run_metadata(metadata, output_root)
    run_dir = metadata_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    backend_config, scheduler = _build_aggregated_components(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=prompt_tokens + max_new_tokens,
        allow_simulated_backend=allow_simulated_backend,
    )

    started_at = _utc_now()
    wall_start = perf_counter()
    latency_collector = LatencyStatsCollector()
    gpu_collector = GPUStatsCollector(
        sample_interval_ms=_safe_int(serving_config.get("gpu_sample_interval_ms"), 100)
    )
    kv_transfer_collector = KVTransferStatsCollector()
    kv_transfer_collector.set_mode("aggregated")
    latency_collector.start()
    gpu_collector.start()
    kv_transfer_collector.start()

    backend_modes: dict[str, str] = {}
    request_records: list[dict[str, Any]] = []
    concurrency_results: list[dict[str, Any]] = []
    success = False
    error_message: str | None = None
    backend: VLLMBackend | None = None
    try:
        backend = VLLMBackend(backend_config)
        backend.start()
        backend_modes["aggregated"] = backend.mode
        if warmup:
            backend.warmup()

        for concurrency in concurrency_levels:
            total_wave_wall_ms = 0.0
            total_completion_tokens = 0
            total_requests = 0
            goodput_request_count = 0
            level_ttft_values: list[float] = []
            level_tpot_values: list[float] = []

            for wave_idx in range(waves_per_level):
                wave_requests: list[GenerationRequest] = []
                for slot in range(concurrency):
                    request_id = (
                        f"{experiment_name}-c{concurrency}-w{wave_idx + 1:03d}-r{slot + 1:03d}"
                    )
                    request = _build_generation_request(
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        experiment_name=experiment_name,
                    )
                    wave_requests.append(request)

                wave_started = perf_counter()
                for request in wave_requests:
                    scheduler.route_prefill(request)
                    backend.prefill(request)
                    scheduler.route_decode(request.request_id or "")
                if hasattr(backend, "generate_batch"):
                    generation_results = backend.generate_batch(wave_requests)
                else:
                    generation_results = [backend.generate(request) for request in wave_requests]
                wave_wall_ms = (perf_counter() - wave_started) * 1000.0
                total_wave_wall_ms += wave_wall_ms

                for request, generation_result in zip(wave_requests, generation_results):
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
                        request_id=request.request_id or "e2-req-unknown",
                        ttft_ms=generation_result.ttft_ms,
                        tpot_ms=generation_result.tpot_ms,
                        prompt_tokens=generation_result.prompt_tokens,
                        completion_tokens=generation_result.completion_tokens,
                        metadata={
                            **generation_result.metadata,
                            "concurrency_level": concurrency,
                            "wave_index": wave_idx + 1,
                        },
                    )
                    request_records.append(
                        {
                            "request_id": request.request_id,
                            "concurrency": concurrency,
                            "wave_index": wave_idx + 1,
                            "prompt_tokens_observed": generation_result.prompt_tokens,
                            "completion_tokens": generation_result.completion_tokens,
                            "ttft_ms": generation_result.ttft_ms,
                            "tpot_ms": generation_result.tpot_ms,
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
                    "concurrency": concurrency,
                    "waves": waves_per_level,
                    "request_count": total_requests,
                    "total_completion_tokens": total_completion_tokens,
                    "total_wall_ms": total_wave_wall_ms,
                    "throughput_tokens_per_s": throughput_tokens_per_s,
                    "goodput_rps": goodput_rps,
                    "goodput_request_count": goodput_request_count,
                    "ttft_ms": _distribution(level_ttft_values),
                    "tpot_ms": _distribution(level_tpot_values),
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
    finished_at = _utc_now()
    kv_transfer_collector.set_wall_time_ms(wall_time_ms)
    profiling_snapshot = {
        "latency": latency_collector.snapshot(),
        "gpu": gpu_collector.snapshot(),
        "kv_transfer": kv_transfer_collector.snapshot(),
    }

    csv_rows = [
        {
            "concurrency": result["concurrency"],
            "waves": result["waves"],
            "request_count": result["request_count"],
            "total_completion_tokens": result["total_completion_tokens"],
            "total_wall_ms": result["total_wall_ms"],
            "throughput_tokens_per_s": result["throughput_tokens_per_s"],
            "goodput_rps": result["goodput_rps"],
            "ttft_p95_ms": result["ttft_ms"]["p95"],
            "tpot_p95_ms": result["tpot_ms"]["p95"],
        }
        for result in concurrency_results
    ]
    e2_csv_path = artifacts_dir / "e2_concurrency_curve.csv"
    _write_csv_rows(
        output_path=e2_csv_path,
        fieldnames=[
            "concurrency",
            "waves",
            "request_count",
            "total_completion_tokens",
            "total_wall_ms",
            "throughput_tokens_per_s",
            "goodput_rps",
            "ttft_p95_ms",
            "tpot_p95_ms",
        ],
        rows=csv_rows,
    )

    plot_x = [float(item["concurrency"]) for item in concurrency_results]
    e2_throughput_plot_path = artifacts_dir / "e2_throughput_curve.svg"
    _write_line_chart_svg(
        output_path=e2_throughput_plot_path,
        title=f"{experiment_name}: Throughput/Goodput vs Concurrency",
        x_label="Concurrency Level",
        y_label="Requests/Tokens per Second",
        x_values=plot_x,
        series=[
            ("Throughput tok/s", [float(item["throughput_tokens_per_s"]) for item in concurrency_results]),
            ("Goodput req/s", [float(item["goodput_rps"]) for item in concurrency_results]),
        ],
    )
    e2_latency_plot_path = artifacts_dir / "e2_latency_curve.svg"
    _write_line_chart_svg(
        output_path=e2_latency_plot_path,
        title=f"{experiment_name}: P95 Latency vs Concurrency",
        x_label="Concurrency Level",
        y_label="Latency (ms)",
        x_values=plot_x,
        series=[
            ("TTFT p95", [float(item["ttft_ms"]["p95"] or 0.0) for item in concurrency_results]),
            ("TPOT p95", [float(item["tpot_ms"]["p95"] or 0.0) for item in concurrency_results]),
        ],
    )

    checks = {
        "fixed_seed_configured": True,
        "strict_backend": strict_backend,
        "all_backends_real_vllm": bool(backend_modes)
        and all(mode_name == "vllm" for mode_name in backend_modes.values()),
        "executed_requests": len(request_records),
        "plots_generated": e2_throughput_plot_path.exists() and e2_latency_plot_path.exists(),
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
        "metadata_path": str(metadata_path.resolve()),
        "profiling": profiling_snapshot,
        "route_log": scheduler.route_log(),
        "request_records": request_records,
        "concurrency_results": concurrency_results,
        "slo": {
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
        },
        "artifacts": {
            "e2_concurrency_curve_csv": str(e2_csv_path.resolve()),
            "e2_throughput_curve_svg": str(e2_throughput_plot_path.resolve()),
            "e2_latency_curve_svg": str(e2_latency_plot_path.resolve()),
        },
        "environment_manifest": _env_manifest(),
        "run_metadata": run_metadata_to_dict(metadata),
    }

    summary_path = run_dir / "e2_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return E2RunResult(
        run_id=metadata.run_id,
        run_dir=str(run_dir.resolve()),
        metadata_path=str(metadata_path.resolve()),
        summary_path=str(summary_path.resolve()),
        mode="aggregated",
        success=success,
    )
