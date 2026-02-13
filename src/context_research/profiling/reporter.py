from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_run_dir(target: str, runs_root: str | Path = "outputs/runs") -> Path:
    root = Path(runs_root)
    if target == "latest":
        run_dirs = sorted([path for path in root.glob("run-*") if path.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found under {root}.")
        return run_dirs[-1]

    candidate = Path(target)
    if candidate.is_dir():
        return candidate

    run_dir = root / target
    if run_dir.is_dir():
        return run_dir

    raise FileNotFoundError(f"Unable to resolve run directory from target: {target}")


def resolve_summary_path(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    e0_summary = path / "e0_summary.json"
    if e0_summary.exists():
        return e0_summary

    candidates = sorted(path.glob("*_summary.json"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"No *_summary.json file found under {path}")


def build_run_markdown_report(payload: dict[str, Any], summary_path: str | Path) -> str:
    run_id = str(payload.get("run_id", "unknown-run"))
    mode = payload.get("mode")
    backend = payload.get("backend")
    success = payload.get("success")
    wall_time_ms = payload.get("wall_time_ms")
    started_at = payload.get("started_at_utc")
    finished_at = payload.get("finished_at_utc")
    experiment_name = payload.get("experiment_name")

    generation_result = payload.get("generation_result") or {}
    prefill_result = payload.get("prefill_result") or {}
    checks = payload.get("checks") or {}
    backend_modes = payload.get("backend_modes") or {}
    profiling = payload.get("profiling") or {}
    gpu_assignment = _as_dict(payload.get("gpu_assignment"))
    artifacts = _as_dict(payload.get("artifacts"))
    e1_points = _as_list_of_dicts(payload.get("latency_by_prompt_tokens"))
    e2_points = _as_list_of_dicts(payload.get("concurrency_results"))

    latency_summary = _as_dict(_as_dict(profiling.get("latency")).get("summary"))
    ttft_summary = _as_dict(latency_summary.get("ttft_ms"))
    tpot_summary = _as_dict(latency_summary.get("tpot_ms"))

    gpu_profile = _as_dict(profiling.get("gpu"))
    gpu_run_summary = _as_dict(gpu_profile.get("run_summary"))

    kv_profile = _as_dict(profiling.get("kv_transfer"))
    kv_summary = _as_dict(kv_profile.get("summary"))

    summary_file = Path(summary_path).resolve()
    run_dir = summary_file.parent
    metadata_file = run_dir / "run_metadata.json"

    lines = [
        f"# WP1 Run Summary: {run_id}",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Success: `{success}`",
        f"- Mode: `{mode}`",
        f"- Backend: `{backend}`",
        f"- Backend modes: `{backend_modes}`",
        f"- Started (UTC): `{started_at}`",
        f"- Finished (UTC): `{finished_at}`",
        f"- Wall time (ms): `{_fmt_float(wall_time_ms)}`",
        "",
        "## GPU Assignment",
        "",
        f"- Strategy: `{_fmt_text(gpu_assignment.get('strategy'))}`",
        f"- Tensor parallel size: `{_fmt_int(gpu_assignment.get('tensor_parallel_size'))}`",
        f"- Available GPUs: `{_fmt_list(gpu_assignment.get('available_gpu_indices'))}`",
        f"- Prefill visible devices: `{_fmt_text(gpu_assignment.get('prefill_visible_devices'))}`",
        f"- Decode visible devices: `{_fmt_text(gpu_assignment.get('decode_visible_devices'))}`",
        f"- Warning: `{_fmt_text(gpu_assignment.get('warning'))}`",
        "",
        "## Request Latency",
        "",
        f"- Request count: `{_fmt_int(_as_dict(profiling.get('latency')).get('sample_count'))}`",
        f"- TTFT p50/p95 (ms): `{_fmt_float(ttft_summary.get('p50'))}` / `{_fmt_float(ttft_summary.get('p95'))}`",
        f"- TPOT p50/p95 (ms): `{_fmt_float(tpot_summary.get('p50'))}` / `{_fmt_float(tpot_summary.get('p95'))}`",
        f"- Last request TTFT/TPOT (ms): `{_fmt_float(generation_result.get('ttft_ms'))}` / `{_fmt_float(generation_result.get('tpot_ms'))}`",
        f"- Prompt/completion tokens: `{_fmt_int(generation_result.get('prompt_tokens'))}` / `{_fmt_int(generation_result.get('completion_tokens'))}`",
        f"- Latency source: `{_fmt_text(_as_dict(generation_result.get('metadata')).get('latency_source'))}`",
    ]

    if e1_points:
        lines.extend(
            [
                "",
                "## E1 Sweep",
                "",
                "| Prompt Tokens | Trials | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) |",
                "| :---- | :---- | :---- | :---- | :---- | :---- |",
            ]
        )
        for point in e1_points:
            ttft = _as_dict(point.get("ttft_ms"))
            tpot = _as_dict(point.get("tpot_ms"))
            lines.append(
                "| "
                f"{_fmt_int(point.get('prompt_tokens'))} | "
                f"{_fmt_int(point.get('trial_count'))} | "
                f"{_fmt_float(ttft.get('p50'))} | "
                f"{_fmt_float(ttft.get('p95'))} | "
                f"{_fmt_float(tpot.get('p50'))} | "
                f"{_fmt_float(tpot.get('p95'))} |"
            )
        lines.extend(
            [
                "",
                f"- E1 CSV: `{_fmt_text(artifacts.get('e1_latency_curve_csv'))}`",
                f"- E1 TTFT plot: `{_fmt_text(artifacts.get('e1_ttft_curve_svg'))}`",
                f"- E1 TPOT plot: `{_fmt_text(artifacts.get('e1_tpot_curve_svg'))}`",
            ]
        )

    if e2_points:
        lines.extend(
            [
                "",
                "## E2 Sweep",
                "",
                "| Concurrency | Requests | Throughput (tok/s) | Goodput (req/s) | TTFT p95 (ms) | TPOT p95 (ms) |",
                "| :---- | :---- | :---- | :---- | :---- | :---- |",
            ]
        )
        for point in e2_points:
            ttft = _as_dict(point.get("ttft_ms"))
            tpot = _as_dict(point.get("tpot_ms"))
            lines.append(
                "| "
                f"{_fmt_int(point.get('concurrency'))} | "
                f"{_fmt_int(point.get('request_count'))} | "
                f"{_fmt_float(point.get('throughput_tokens_per_s'))} | "
                f"{_fmt_float(point.get('goodput_rps'))} | "
                f"{_fmt_float(ttft.get('p95'))} | "
                f"{_fmt_float(tpot.get('p95'))} |"
            )
        lines.extend(
            [
                "",
                f"- E2 CSV: `{_fmt_text(artifacts.get('e2_concurrency_curve_csv'))}`",
                f"- E2 throughput plot: `{_fmt_text(artifacts.get('e2_throughput_curve_svg'))}`",
                f"- E2 latency plot: `{_fmt_text(artifacts.get('e2_latency_curve_svg'))}`",
            ]
        )

    lines.extend(
        [
        "",
        "## Prefill",
        "",
        f"- Prefill ms: `{_fmt_float(prefill_result.get('prefill_ms'))}`",
        f"- Prefill tokens: `{_fmt_int(prefill_result.get('prompt_tokens'))}`",
        "",
        "## GPU Stats",
        "",
        f"- Collector available: `{_fmt_bool(gpu_profile.get('available'))}`",
        f"- GPU samples: `{_fmt_int(gpu_profile.get('sample_count'))}`",
        f"- Memory used avg/peak (MB): `{_fmt_float(gpu_run_summary.get('memory_used_mb_avg'))}` / `{_fmt_float(gpu_run_summary.get('memory_used_mb_peak'))}`",
        f"- GPU util avg/peak (%): `{_fmt_float(gpu_run_summary.get('gpu_utilization_pct_avg'))}` / `{_fmt_float(gpu_run_summary.get('gpu_utilization_pct_peak'))}`",
        "",
        "## KV Transfer (Disaggregated)",
        "",
        f"- Collector available: `{_fmt_bool(kv_profile.get('available'))}`",
        f"- Completed handoffs: `{_fmt_int(kv_profile.get('sample_count'))}`",
        f"- Total transfer bytes: `{_fmt_int(kv_summary.get('total_transfer_bytes'))}`",
        f"- Total transfer ms: `{_fmt_float(kv_summary.get('total_transfer_ms'))}`",
        f"- Throughput (MB/s): `{_fmt_float(kv_summary.get('transfer_throughput_mb_per_s'))}`",
        f"- Stall ratio: `{_fmt_float(kv_summary.get('stall_ratio'))}`",
        f"- Stall ratio basis: `{_fmt_text(kv_summary.get('stall_ratio_basis'))}`",
        "",
        "## Checks",
        "",
        f"- {checks}",
        "",
        "## Files",
        "",
        f"- Summary JSON: `{summary_file}`",
        f"- Metadata JSON: `{metadata_file.resolve()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_run_report(
    run_target: str,
    *,
    runs_root: str | Path = "outputs/runs",
    reports_root: str | Path = "outputs/reports",
) -> Path:
    run_dir = resolve_run_dir(run_target, runs_root=runs_root)
    summary_path = resolve_summary_path(run_dir)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    report_text = build_run_markdown_report(payload, summary_path)

    run_id = str(payload.get("run_id", run_dir.name))
    output_dir = Path(reports_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_id}.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path.resolve()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _fmt_int(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value))
    return "n/a"


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    return "n/a"


def _fmt_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "n/a"


def _fmt_list(value: Any) -> str:
    if isinstance(value, list):
        return str(value)
    return "n/a"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-research-report",
        description="Generate a markdown report for a WP1 run directory.",
    )
    parser.add_argument(
        "run_target",
        nargs="?",
        default="latest",
        help="Run ID, run directory path, or 'latest'.",
    )
    parser.add_argument(
        "--runs-root",
        default="outputs/runs",
        help="Root directory containing run-* folders.",
    )
    parser.add_argument(
        "--reports-root",
        default="outputs/reports",
        help="Output directory for markdown report files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report_path = generate_run_report(
        args.run_target,
        runs_root=args.runs_root,
        reports_root=args.reports_root,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
