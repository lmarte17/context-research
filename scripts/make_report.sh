#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TARGET="${1:-latest}"

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" python - "${RUN_TARGET}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys


def resolve_run_dir(target: str) -> Path:
    runs_root = Path("outputs/runs")
    if target == "latest":
        run_dirs = sorted([p for p in runs_root.glob("run-*") if p.is_dir()])
        if not run_dirs:
            raise SystemExit("No run directories found under outputs/runs.")
        return run_dirs[-1]

    candidate = Path(target)
    if candidate.is_dir():
        return candidate

    run_dir = runs_root / target
    if run_dir.is_dir():
        return run_dir

    raise SystemExit(f"Unable to resolve run directory from target: {target}")


def main() -> int:
    run_target = sys.argv[1]
    run_dir = resolve_run_dir(run_target)
    summary_path = run_dir / "e0_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = payload["run_id"]
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.md"

    generation_result = payload.get("generation_result") or {}
    prefill_result = payload.get("prefill_result") or {}
    checks = payload.get("checks") or {}
    backend_modes = payload.get("backend_modes") or {}

    lines = [
        f"# E0 Summary: {run_id}",
        "",
        f"- Success: `{payload.get('success')}`",
        f"- Mode: `{payload.get('mode')}`",
        f"- Backend: `{payload.get('backend')}`",
        f"- Backend modes: `{backend_modes}`",
        f"- Seed: `{payload.get('seed')}`",
        f"- Wall time (ms): `{payload.get('wall_time_ms')}`",
        "",
        "## Latency",
        "",
        f"- TTFT ms: `{generation_result.get('ttft_ms')}`",
        f"- TPOT ms: `{generation_result.get('tpot_ms')}`",
        f"- Prompt tokens: `{generation_result.get('prompt_tokens')}`",
        f"- Completion tokens: `{generation_result.get('completion_tokens')}`",
        f"- Latency source: `{(generation_result.get('metadata') or {}).get('latency_source')}`",
        "",
        "## Prefill",
        "",
        f"- Prefill ms: `{prefill_result.get('prefill_ms')}`",
        f"- Prefill tokens: `{prefill_result.get('prompt_tokens')}`",
        "",
        "## Checks",
        "",
        f"- {checks}",
        "",
        "## Files",
        "",
        f"- Summary JSON: `{summary_path.resolve()}`",
        f"- Metadata JSON: `{(run_dir / 'run_metadata.json').resolve()}`",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path.resolve())
    return 0


raise SystemExit(main())
PY
