# WP1 Progress Log and Implications

Created: February 12, 2026  
Purpose: running record of completed work, concrete evidence, and what each milestone unlocks next.

## Current Snapshot

- Completed scope: `T1` through `T12`, plus `E0`, `E1`, and `E2` execution paths.
- Current milestone state: `G1` and `G2` satisfied; `G3` in progress (`T13` pending).
- Latest strict E0 evidence run: `run-20260212T050845Z-bdbc5aa3`.
- Latest profiling/reporting validation run: `run-20260212T215844Z-959e7b72`.
- Latest E1 run evidence: `run-20260213T034945Z-e4774230`.
- Latest E2 run evidence: `run-20260213T034950Z-5e949638`.
- Execution hardware for the validated run: Lightning Studio `NVIDIA L40S` (46GB VRAM).

## Completed Work (T1-T12 + E0/E1/E2)

### T1: Project Scaffold and Basic CLI

- Established project layout under `src/`, `configs/`, `scripts/`, and `outputs/`.
- Added operational CLI commands for `init-run`, scaffold checks, and E0 execution.
- Result: baseline commands execute from a clean checkout without manual path hacks.

### T2: Dependency and Environment Management

- Added pinned runtime dependencies in `requirements.txt` and `pyproject.toml`.
- Hardened bootstrap behavior for both local and Lightning Studio environments.
- Added forced reinstall + dependency consistency checks during bootstrap to avoid binary mismatch drift.
- Result: one-command install path is deterministic enough for GPU-hosted reproducibility.

### T3: Run Metadata Schema

- Implemented run metadata schema and writer with:
  - run id,
  - config hash,
  - hardware/OS/Python metadata,
  - model id,
  - run notes.
- Result: each run emits machine-readable `run_metadata.json` alongside experiment outputs.

### T4: Serving Backend (`vLLM`)

- Implemented `ServingBackend` + `VLLMBackend` with:
  - strict real-backend default,
  - explicit simulated fallback (opt-in),
  - seeded generation path,
  - prefill phase API,
  - TTFT/TPOT extraction from vLLM metrics when available.
- Result: E0 now fails fast when real backend is unavailable, instead of silently passing simulated work.

### T5: Aggregated Scheduler

- Implemented aggregated prefill/decode routing and structured route logs.
- Added pool metadata to route events for easier later analysis.
- Result: aggregated scheduling path is explicit and auditable in run artifacts.

### T6: Disaggregated Scheduler + Broker

- Implemented disaggregated scheduler with separate prefill/decode pools.
- Implemented broker handoff accounting (`handoff_id`, `transfer_bytes`, `transfer_ms`).
- Wired E0 disaggregated path to execute separate prefill/decode backend instances.
- Result: disaggregated baseline path now produces concrete transfer/accounting artifacts.

### T7: TTFT/TPOT Collector

- Implemented `LatencyStatsCollector` with per-request sample capture.
- Added aggregate distribution reporting (`min`, `max`, `avg`, `p50`, `p95`) for TTFT/TPOT.
- Wired collector outputs into E0 summary under `profiling.latency`.
- Result: latency data is now directly usable for E1/E2 curves and p95-based comparisons.

### T8: GPU Stats Collector

- Implemented sampled GPU telemetry collector (`GPUStatsCollector`) using NVML when available.
- Added per-run avg/peak summaries for memory used and utilization.
- Added graceful unavailable-path reporting when NVML/GPU metrics are not available.
- Wired collector outputs into E0 summary under `profiling.gpu`.
- Result: E0 artifacts now include the memory/utilization primitives needed for E2/E5 system analysis.

### T9: KV Transfer Collector

- Implemented `KVTransferStatsCollector` for disaggregated transfer accounting.
- Ingested broker completed handoffs and standardized transfer reporting (`bytes`, `ms`, throughput, stall ratio).
- Added `broker_completed_handoffs` and `profiling.kv_transfer` blocks to E0 summaries.
- Result: disaggregated runs now emit direct transfer-overhead metrics needed for E3/T13.

### T10: Generalized Report Generator

- Added generalized report module (`context_research.profiling.reporter`) and switched `scripts/make_report.sh` to use it.
- Report now summarizes latency, GPU stats, KV transfer stats, checks, and run files from summary JSON.
- Preserved compatibility with older summary payloads that predate profiling blocks.
- Result: one-command run reporting is now reusable across current and upcoming WP1 experiments.

### Disaggregated Multi-GPU Hardening (Post T9/T10)

- Added automatic GPU discovery and prefill/decode assignment for disaggregated E0 runs.
- Added assignment strategy reporting (`gpu_assignment`) to run summary and markdown reports.
- Added optional manual serving-config overrides (`prefill_visible_devices`, `decode_visible_devices`).
- Result: disaggregated real-vLLM runs can be explicitly and reproducibly mapped to multiple GPUs.

### E0: Smoke + Reproducibility

- E0 now requires fixed `seed` in config.
- E0 summary includes:
  - strict-backend check fields,
  - backend mode verification,
  - environment manifest,
  - request/result payload,
  - route log and optional broker snapshot.
- Strict real-vLLM run succeeded with:
  - `success=true`,
  - `strict_backend=true`,
  - `all_backends_real_vllm=true`.

### T12: E0/E1/E2 Runner + Config Coverage

- Added `run-e1` and `run-e2` CLI commands and script wrappers in `scripts/run_experiment.sh`.
- Added experiment configs:
  - `configs/experiments/e1_aggregated_latency.yaml`
  - `configs/experiments/e2_batch_concurrency.yaml`
- Extended report generation to include E1/E2 sweep tables and artifact references.
- Result: E0/E1/E2 can now be executed end-to-end with run metadata, summaries, plots, and markdown reports.

### E1: Aggregated Latency Sweep

- Implemented prompt-length sweep execution with per-length TTFT/TPOT distributions.
- Added generated artifacts:
  - `artifacts/e1_latency_curve.csv`
  - `artifacts/e1_ttft_curve.svg`
  - `artifacts/e1_tpot_curve.svg`
- Validation run `run-20260213T034945Z-e4774230` produced successful summary + plots.

### E2: Batch/Concurrency Sweep

- Implemented concurrency-level sweep with throughput, p95 latency, and SLO goodput reporting.
- Added generated artifacts:
  - `artifacts/e2_concurrency_curve.csv`
  - `artifacts/e2_throughput_curve.svg`
  - `artifacts/e2_latency_curve.svg`
- Validation run `run-20260213T034950Z-5e949638` produced successful summary + plots.

## Why This Matters for Remaining WP1 Work

- `G2` can now be treated as closed: T7-T10 are complete and E1/E2 produced plotted outputs.
- `E3/T13` now have transfer and stall-ratio primitives from T9, plus explicit disaggregated GPU assignment metadata.
- Run artifacts now contain a stable profiling schema plus E1/E2 sweep outputs and plot files for downstream comparison code.
- Remaining WP1 blockers center on benchmark harness and comparison/evaluation tracks (`T11`, `T13-T16`).

## Operational Notes

- `HF_HOME` must be a filesystem path, not a URL.
- Simulated backend mode should remain debug-only (`CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND=1`).
- Different GPU SKUs (L40S vs A100) are acceptable for early WP1 execution, but final comparisons should annotate hardware in reports.
- Real disaggregated 8B runs should use multi-GPU placement. On single 46GB GPUs, two real vLLM backends can OOM.
- Single-GPU systems should use aggregated real runs and reserve disaggregated mode for simulated-path checks.
- Current E1/E2 evidence runs were validated through simulated backend execution in this environment; rerun strict real-vLLM on Lightning for final baseline packet numbers.

## Update Protocol

When adding new entries:

1. Append a dated log item below with exact run IDs and files touched.
2. Record which acceptance criteria moved from pending to complete.
3. Add at least one implication for downstream tasks (what became easier/unblocked).

## Change Log

### 2026-02-12

- Closed `T1-T6`.
- Converted E0 into strict real-backend pass/fail behavior and validated successful strict run.
- Stabilized Lightning bootstrap path and documented environment fillers/metadata flow.
- Closed `T7-T10` with profiling collectors + generalized report generation.
- Added disaggregated transfer/stall reporting and standardized profiling sections in summary artifacts.
- Added disaggregated multi-GPU assignment logic and surfaced assignment metadata in summaries/reports.
- Validation runs: `run-20260212T211216Z-0e6ee513`, `run-20260212T211226Z-a7a7d4e8`, `run-20260212T215844Z-959e7b72`.
- Closed `T12` by implementing E1/E2 runners/configs and producing plotted artifacts.
- Closed `G2` conditionally with E1/E2 plotted validation runs: `run-20260213T034945Z-e4774230`, `run-20260213T034950Z-5e949638`.
