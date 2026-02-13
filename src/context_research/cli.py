from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from context_research.config.schema import create_run_metadata, write_run_metadata
from context_research.experiments.runner import run_e0, run_e1, run_e2
from context_research.experiments.extended_runner import run_e3, run_e4, run_e5, run_e6


def _parse_notes(notes_json: str | None) -> dict[str, object]:
    if not notes_json:
        return {}
    parsed = json.loads(notes_json)
    if not isinstance(parsed, dict):
        raise ValueError("--notes-json must decode to a JSON object.")
    return parsed


def cmd_init_run(args: argparse.Namespace) -> int:
    notes = _parse_notes(args.notes_json)
    metadata = create_run_metadata(
        model_name=args.model_name,
        config_path=args.config_path,
        run_id=args.run_id,
        notes=notes,
    )
    output_path = write_run_metadata(metadata, args.output_root)
    print(output_path)
    return 0


def cmd_scaffold_dirs(_: argparse.Namespace) -> int:
    base_dirs = [
        Path("src/context_research"),
        Path("configs/model"),
        Path("configs/serving"),
        Path("configs/benchmarks"),
        Path("configs/experiments"),
        Path("scripts"),
        Path("data/raw"),
        Path("data/processed"),
        Path("outputs/runs"),
        Path("outputs/reports"),
    ]
    for directory in base_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    print("Scaffold directories are present.")
    return 0


def cmd_run_e0(args: argparse.Namespace) -> int:
    result = run_e0(
        experiment_config_path=args.experiment_config,
        serving_config_path=args.serving_config,
        output_root=args.output_root,
        run_id=args.run_id,
        request_prompt=args.prompt,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def cmd_run_e1(args: argparse.Namespace) -> int:
    result = run_e1(
        experiment_config_path=args.experiment_config,
        serving_config_path=args.serving_config,
        output_root=args.output_root,
        run_id=args.run_id,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def cmd_run_e2(args: argparse.Namespace) -> int:
    result = run_e2(
        experiment_config_path=args.experiment_config,
        serving_config_path=args.serving_config,
        output_root=args.output_root,
        run_id=args.run_id,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def cmd_run_e3(args: argparse.Namespace) -> int:
    result = run_e3(
        experiment_config_path=args.experiment_config,
        output_root=args.output_root,
        run_id=args.run_id,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
        aggregated_serving_config_path=args.aggregated_serving_config,
        disaggregated_serving_config_path=args.disaggregated_serving_config,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def cmd_run_e4(args: argparse.Namespace) -> int:
    result = run_e4(
        experiment_config_path=args.experiment_config,
        serving_config_path=args.serving_config,
        output_root=args.output_root,
        run_id=args.run_id,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
        benchmark_config_path=args.benchmark_config,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def cmd_run_e5(args: argparse.Namespace) -> int:
    result = run_e5(
        experiment_config_path=args.experiment_config,
        output_root=args.output_root,
        run_id=args.run_id,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
        aggregated_serving_config_path=args.aggregated_serving_config,
        disaggregated_serving_config_path=args.disaggregated_serving_config,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def cmd_run_e6(args: argparse.Namespace) -> int:
    result = run_e6(
        experiment_config_path=args.experiment_config,
        serving_config_path=args.serving_config,
        output_root=args.output_root,
        run_id=args.run_id,
        warmup=not args.no_warmup,
        strict_backend=not args.allow_simulated_backend,
        allow_simulated_backend=args.allow_simulated_backend,
        benchmark_config_path=args.benchmark_config,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "metadata_path": result.metadata_path,
                "summary_path": result.summary_path,
                "mode": result.mode,
                "success": result.success,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.success else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="context-research",
        description="WP1 scaffolding utilities for context-research.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_run = subparsers.add_parser(
        "init-run",
        help="Create run metadata JSON with run_id, hardware, model, and config hash.",
    )
    init_run.add_argument(
        "--model-name",
        required=True,
        help="Model identifier, e.g. Qwen/Qwen3-8B.",
    )
    init_run.add_argument(
        "--config-path",
        required=True,
        help="Path to the experiment config file used for hashing.",
    )
    init_run.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/run_metadata.json will be written.",
    )
    init_run.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    init_run.add_argument(
        "--notes-json",
        default=None,
        help="Optional JSON object string to store extra metadata.",
    )
    init_run.set_defaults(func=cmd_init_run)

    scaffold = subparsers.add_parser(
        "scaffold-dirs",
        help="Ensure the WP1 baseline folder structure exists.",
    )
    scaffold.set_defaults(func=cmd_scaffold_dirs)

    run_e0_parser = subparsers.add_parser(
        "run-e0",
        help="Execute E0 smoke experiment end-to-end and write run artifacts.",
    )
    run_e0_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e0_smoke.yaml",
        help="Path to E0 experiment YAML config.",
    )
    run_e0_parser.add_argument(
        "--serving-config",
        default="configs/serving/aggregated.yaml",
        help="Path to serving YAML config (aggregated or disaggregated).",
    )
    run_e0_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e0_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e0_parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt override for smoke generation.",
    )
    run_e0_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before the measured generation.",
    )
    run_e0_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e0_parser.set_defaults(func=cmd_run_e0)

    run_e1_parser = subparsers.add_parser(
        "run-e1",
        help="Execute E1 aggregated latency sweep and write run artifacts.",
    )
    run_e1_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e1_aggregated_latency.yaml",
        help="Path to E1 experiment YAML config.",
    )
    run_e1_parser.add_argument(
        "--serving-config",
        default="configs/serving/aggregated.yaml",
        help="Path to serving YAML config (aggregated).",
    )
    run_e1_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e1_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e1_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before the sweep.",
    )
    run_e1_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e1_parser.set_defaults(func=cmd_run_e1)

    run_e2_parser = subparsers.add_parser(
        "run-e2",
        help="Execute E2 batch/concurrency sweep and write run artifacts.",
    )
    run_e2_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e2_batch_concurrency.yaml",
        help="Path to E2 experiment YAML config.",
    )
    run_e2_parser.add_argument(
        "--serving-config",
        default="configs/serving/aggregated.yaml",
        help="Path to serving YAML config (aggregated).",
    )
    run_e2_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e2_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e2_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before the sweep.",
    )
    run_e2_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e2_parser.set_defaults(func=cmd_run_e2)

    run_e3_parser = subparsers.add_parser(
        "run-e3",
        help="Execute E3 aggregated vs disaggregated comparison and write run artifacts.",
    )
    run_e3_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e3_disaggregated_vs_aggregated.yaml",
        help="Path to E3 experiment YAML config.",
    )
    run_e3_parser.add_argument(
        "--aggregated-serving-config",
        default=None,
        help="Optional override for aggregated serving YAML config.",
    )
    run_e3_parser.add_argument(
        "--disaggregated-serving-config",
        default=None,
        help="Optional override for disaggregated serving YAML config.",
    )
    run_e3_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e3_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e3_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before the comparison sweeps.",
    )
    run_e3_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e3_parser.set_defaults(func=cmd_run_e3)

    run_e4_parser = subparsers.add_parser(
        "run-e4",
        help="Execute E4 capability subset evaluation and write run artifacts.",
    )
    run_e4_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e4_capability_subset.yaml",
        help="Path to E4 experiment YAML config.",
    )
    run_e4_parser.add_argument(
        "--serving-config",
        default="configs/serving/aggregated.yaml",
        help="Path to serving YAML config (aggregated).",
    )
    run_e4_parser.add_argument(
        "--benchmark-config",
        default=None,
        help="Optional override for benchmark suite YAML config.",
    )
    run_e4_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e4_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e4_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before evaluation.",
    )
    run_e4_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e4_parser.set_defaults(func=cmd_run_e4)

    run_e5_parser = subparsers.add_parser(
        "run-e5",
        help="Execute E5 memory decomposition sweep and write run artifacts.",
    )
    run_e5_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e5_memory_decomposition.yaml",
        help="Path to E5 experiment YAML config.",
    )
    run_e5_parser.add_argument(
        "--aggregated-serving-config",
        default=None,
        help="Optional override for aggregated serving YAML config.",
    )
    run_e5_parser.add_argument(
        "--disaggregated-serving-config",
        default=None,
        help="Optional override for disaggregated serving YAML config.",
    )
    run_e5_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e5_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e5_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before each memory case.",
    )
    run_e5_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e5_parser.set_defaults(func=cmd_run_e5)

    run_e6_parser = subparsers.add_parser(
        "run-e6",
        help="Execute E6 YaRN stress subset evaluation and write run artifacts.",
    )
    run_e6_parser.add_argument(
        "--experiment-config",
        default="configs/experiments/e6_yarn_stress.yaml",
        help="Path to E6 experiment YAML config.",
    )
    run_e6_parser.add_argument(
        "--serving-config",
        default="configs/serving/aggregated.yaml",
        help="Path to serving YAML config (aggregated).",
    )
    run_e6_parser.add_argument(
        "--benchmark-config",
        default=None,
        help="Optional override for benchmark suite YAML config.",
    )
    run_e6_parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Directory where outputs/runs/<run_id>/ artifacts will be written.",
    )
    run_e6_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id; autogenerated when omitted.",
    )
    run_e6_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip backend warmup request before each target length.",
    )
    run_e6_parser.add_argument(
        "--allow-simulated-backend",
        action="store_true",
        help=(
            "Allow simulated fallback when vLLM is unavailable. "
            "Default behavior is strict real-backend execution."
        ),
    )
    run_e6_parser.set_defaults(func=cmd_run_e6)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
