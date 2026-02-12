# WP1 Progress Log and Implications

Created: February 12, 2026  
Purpose: running record of completed work, concrete evidence, and what each milestone unlocks next.

## Current Snapshot

- Completed scope: `T1` through `T6`, plus `E0` smoke/reproducibility run.
- Current milestone state: `G1` satisfied.
- Latest strict E0 evidence run: `run-20260212T050845Z-bdbc5aa3`.
- Execution hardware for the validated run: Lightning Studio `NVIDIA L40S` (46GB VRAM).

## Completed Work (T1-T6 + E0)

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

## Why This Matters for Remaining WP1 Work

- `T7` (latency collector): base per-request fields already exist; remaining work is aggregate statistics across batches/runs (p50/p95).
- `T8` (GPU collector): current manifest confirms GPU identity; remaining work is time-series sampling and per-run summary statistics.
- `T9` (KV transfer collector): transfer bytes/time primitives already exist in broker/scheduler; remaining work is stall-ratio derivation and standardized reporting.
- `T10` (reporting): E0 markdown report path exists; remaining work is generalized multi-experiment report assembly.
- `T11-T16` (benchmarks/experiments): now unblocked by reliable strict-run execution and stable run metadata/output schema.

## Operational Notes

- `HF_HOME` must be a filesystem path, not a URL.
- Simulated backend mode should remain debug-only (`CONTEXT_RESEARCH_ALLOW_SIMULATED_BACKEND=1`).
- Different GPU SKUs (L40S vs A100) are acceptable for early WP1 execution, but final comparisons should annotate hardware in reports.

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
