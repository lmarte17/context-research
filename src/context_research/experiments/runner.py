from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

from context_research.config.io import load_simple_yaml
from context_research.config.schema import create_run_metadata, run_metadata_to_dict, write_run_metadata
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_prompt(max_prompt_tokens: int) -> str:
    target_tokens = max(8, min(max_prompt_tokens, 64))
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

    backend_config = _build_vllm_backend_config(
        model_name=model_name,
        serving_config=serving_config,
        max_model_len=max_prompt_tokens + max_new_tokens,
        allow_simulated_backend=allow_simulated_backend,
    )

    if mode == "aggregated":
        scheduler: AggregatedScheduler | DisaggregatedScheduler = AggregatedScheduler(
            pool_name=_safe_str(serving_config.get("pool_name"), "aggregated"),
            pool_size=_safe_int(serving_config.get("pool_size"), 1),
        )
    else:
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

    generation_result = None
    prefill_result: dict[str, Any] | None = None
    prefill_pool = None
    decode_pool = None
    broker_snapshot: dict[str, Any] | None = None
    success = False
    error_message = None
    backend_modes: dict[str, str] = {}

    aggregated_backend: VLLMBackend | None = None
    prefill_backend: VLLMBackend | None = None
    decode_backend: VLLMBackend | None = None

    try:
        if mode == "aggregated":
            aggregated_backend = VLLMBackend(backend_config)
            aggregated_backend.start()
            if warmup:
                aggregated_backend.warmup()
            prefill_pool = scheduler.route_prefill(request)
            prefill_result = aggregated_backend.prefill(request)
            decode_pool = scheduler.route_decode(request.request_id or "")
            generation_result = aggregated_backend.generate(request)
            backend_modes["aggregated"] = aggregated_backend.mode
        else:
            prefill_backend = VLLMBackend(backend_config)
            decode_backend = VLLMBackend(backend_config)
            prefill_backend.start()
            decode_backend.start()
            if warmup:
                prefill_backend.warmup()
                decode_backend.warmup()
            prefill_pool = scheduler.route_prefill(request)
            prefill_result = prefill_backend.prefill(request)
            decode_pool = scheduler.route_decode(request.request_id or "")
            generation_result = decode_backend.generate(request)
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

    wall_time_ms = (perf_counter() - wall_start) * 1000.0
    finished_at = _utc_now()

    route_log = scheduler.route_log()
    if isinstance(scheduler, DisaggregatedScheduler):
        broker_snapshot = scheduler.broker.snapshot()

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
        "broker_snapshot": broker_snapshot,
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
