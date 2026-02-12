# WP1 Implementation Roadmap (Budget-Constrained 7B-8B Track)

Prepared: February 9, 2026  
Updated: February 12, 2026  
Scope: WP1 only (Baselines and Instrumentation)

## 0. Execution Status (As of February 12, 2026)

| Item | Status | Evidence / Notes |
| :---- | :---- | :---- |
| T1-T3 | complete | scaffold, pinned dependency path, and run metadata schema are in production use |
| T4-T6 | complete | strict real-vLLM backend, aggregated scheduler, and disaggregated scheduler + broker are implemented |
| T7-T10 | complete | latency/GPU/KV transfer collectors and generalized reporting are implemented and wired into E0 artifacts |
| E0 | complete | strict run succeeded (`run-20260212T050845Z-bdbc5aa3`) with `backend_modes.aggregated=vllm` |
| G1 | complete | T1-T5 complete and E0 pass criteria satisfied |
| G2 | in progress | collector/report prerequisites satisfied; pending E1/E2 execution and plot generation |
| T11-T16 | pending / in progress | benchmark harness and downstream experiment tracks remain to be implemented |

## 1. WP1 Objective

Build a reproducible baseline suite for long-context serving that can run on limited hardware and produce credible metrics for later WP2/WP3 comparisons.

WP1 outputs:

- aggregated serving baseline,
- disaggregated serving baseline (prototype-level if needed),
- benchmark harness (capability + systems),
- profiling pipeline (memory/latency/goodput/KV transfer),
- repeatable experiment configs and report generation.

## 2. Budget-First Assumptions

Default assumptions for low-cost setup:

- 1x to 2x consumer GPUs (24GB each) or small rented equivalent.
- 7B to 8B instruct model family only.
- FP16/BF16 when possible; quantized inference where needed.
- start with subset benchmarks, then scale sample counts.

### 2.1 Locked Configuration (Selected)

The following defaults are now selected for WP1 execution:

- **Primary model:** `Qwen/Qwen3-8B`
- **Model policy:** pretrained-first, frozen base model (no full pretraining)
- **Serving stack:** `vLLM` as the only backend in WP1
- **Hardware target:** single 24GB GPU first; optional scale-up later
- **Execution validation hardware (current):** Lightning Studio `NVIDIA L40S` (46GB)
- **Benchmark depth:** fast fixed subsets for WP1
- **Disaggregation timing:** aggregated baseline first, disaggregated prototype in Month 2
- **Disaggregated execution policy:** real 8B disaggregated runs are multi-GPU; single-GPU uses aggregated real runs or simulated disaggregated checks
- **Precision strategy:** BF16 where feasible; quantized fallback for longer contexts
- **Context regime:** native 32K first; then targeted 131K YaRN stress subset
- **Thinking mode:** `enable_thinking=False` for latency/system fairness in baseline runs

## 3. Recommended Baseline Stack

### 3.1 Serving Backend

Primary recommendation:

- `vLLM` for aggregated baseline + fast iteration.

Disaggregated prototype recommendation:

- two independent serving pools (`prefill` and `decode`) with a lightweight broker.
- use queue-based handoff and explicit transfer accounting first; optimize later.

Fallback if engineering capacity is tight:

- aggregated-only in Month 1, then emulated disaggregation in Month 2.

### 3.2 Candidate 7B-8B Models

| Option | Pros | Tradeoffs | Suggested Use |
| :---- | :---- | :---- | :---- |
| Qwen3-8B (`Qwen/Qwen3-8B`) | 8B class quality, Apache-2.0, native 32K context, validated 131K with YaRN, GQA-friendly | more VRAM pressure than 7B; requires current `transformers`/serving stack | default primary baseline |
| Qwen2.5-7B-Instruct | very efficient fallback, mature tooling | older generation vs Qwen3 | low-cost secondary |
| Llama-3.1-8B-Instruct | strong ecosystem and comparability | license constraints, memory heavier than 7B | optional comparison model |
| Mistral-7B-Instruct-v0.3 | lightweight, stable infra support | lower headroom on some long-context tasks | contingency fallback |

### 3.3 Pretrained-First Research Strategy

WP1 and early WP2 should use pretrained models directly. This is valid for long-context systems research because the research object is serving/memory/control behavior under fixed models.

- **No full model pretraining required** for WP1-WP2 baselines.
- Start with **frozen base model** and inference-time methods (paging, scheduling, compression/retention policies).
- If needed later, train only lightweight components (router/adapters) while keeping base weights frozen.
- Report clearly whether gains come from systems policies vs learned adaptation.

## 4. Repository Structure

The tree below is the WP1 target layout. The implemented subset through T1-T10 + E0 is tracked in `docs/wp1/wp1-progress-log.md`.

```text
context-research/
  README.md
  pyproject.toml
  requirements.txt
  .env.example
  docs/
    wp1/
      wp1-implementation-roadmap.md
      wp1-progress-log.md
      wp1-task-backlog.md
    experiment-notes/
  src/
    context_research/
      config/
        schema.py
      serving/
        backends/
          base.py
          vllm_backend.py
        scheduler/
          base.py
          aggregated.py
          disaggregated.py
        broker/
          pd_broker.py
      profiling/
        collectors/
          gpu_stats.py
          latency_stats.py
          kv_transfer_stats.py
        reporter.py
      benchmarks/
        base.py
        pg19.py
        longbench.py
        ruler_subset.py
      experiments/
        runner.py
        matrix.py
      utils/
        logging.py
        io.py
  configs/
    model/
      qwen2_5_7b.yaml
      llama3_1_8b.yaml
    serving/
      aggregated.yaml
      disaggregated.yaml
    benchmarks/
      wp1_smoke.yaml
      wp1_core.yaml
    experiments/
      e0_smoke.yaml
      e1_aggregated_latency.yaml
      e2_batch_concurrency.yaml
      e3_disaggregated_vs_aggregated.yaml
      e4_capability_subset.yaml
  scripts/
    bootstrap.sh
    run_experiment.sh
    make_report.sh
  data/
    raw/
    processed/
  outputs/
    runs/
    reports/
```

## 5. Core Interfaces (WP1)

### 5.1 Serving Backend Interface

```python
from dataclasses import dataclass
from typing import Protocol, Any

@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    request_id: str | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    metadata: dict[str, Any]

class ServingBackend(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def warmup(self) -> None: ...
    def prefill(self, request: GenerationRequest) -> dict[str, Any]: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
```

### 5.2 Scheduler Interface

```python
class Scheduler(Protocol):
    def route_prefill(self, request: GenerationRequest) -> str: ...
    def route_decode(self, request_id: str) -> str: ...
    def route_log(self) -> list[dict[str, Any]]: ...
```

Implementations in WP1:

- `AggregatedScheduler`: prefill and decode on same pool.
- `DisaggregatedScheduler`: separate pools with handoff accounting.

### 5.3 Benchmark Interface

```python
class BenchmarkTask(Protocol):
    name: str
    def load(self, split: str) -> list[dict]: ...
    def evaluate(self, backend: ServingBackend, samples: list[dict]) -> dict: ...
```

### 5.4 Profiling Interface

```python
class MetricCollector(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def snapshot(self) -> dict: ...
```

Required collectors for WP1:

- GPU memory/utilization,
- latency (TTFT/TPOT),
- throughput and goodput,
- KV handoff/transfer bytes and transfer stall ratio.

## 6. Experiment Plan (Initial)

| ID | Purpose | Design | Output | Status | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| E0 | smoke and reproducibility | single model, short prompts, fixed seed | pass/fail + env manifest | complete | strict real-vLLM pass recorded in `run-20260212T050845Z-bdbc5aa3` |
| E1 | baseline latency scaling | aggregated serving; prompt sweep (1K, 4K, 8K, 16K, 32K) | TTFT/TPOT curves | pending | instrumentation/reporting prerequisites are complete; execution + plots pending |
| E2 | throughput and contention | batch/concurrency sweep at fixed prompt lengths | tokens/s, p95 latency, goodput | pending | instrumentation/reporting prerequisites are complete; concurrency harness + runs pending |
| E3 | aggregated vs disaggregated | same workload on both schedulers | delta TTFT/TPOT/goodput and transfer overhead | pending | transfer/stall instrumentation is complete; comparison harness + multi-GPU real runs pending |
| E4 | capability subset | small LongBench + RULER/InfinityBench subset | quality under compute budget | pending | benchmark harness not implemented yet |
| E5 | memory decomposition | profile peak memory across prompt lengths and modes | memory breakdown tables and plots | pending | GPU profiling collector pending |
| E6 | extended-context stress | YaRN-enabled long-context subset (64K, 96K, 131K targets as feasible) | scaling behavior beyond native 32K | pending | requires stable E1/E2 and long-context resource tuning |

Notes:

- if 32K is not feasible, run 24K and document limitation.
- if benchmark runtime is too high, reduce sample count but keep fixed random subset for reproducibility.
- run systems experiments in non-thinking mode for apples-to-apples TTFT/TPOT comparisons.

## 7. 8-Week WP1 Timeline

| Week | Milestone | Status |
| :---- | :---- | :---- |
| 1 | scaffold repo, config schema, backend abstraction, smoke run | complete |
| 2 | aggregated serving runner + TTFT/TPOT instrumentation | complete |
| 3 | profiling collectors + run metadata schema + report template | complete |
| 4 | E1/E2 completed on primary model; baseline dashboard draft | pending |
| 5 | disaggregated scheduler and broker prototype | complete |
| 6 | E3 run + KV transfer accounting; first comparison report | in_progress |
| 7 | capability subset harness (E4) and memory decomposition (E5) | pending |
| 8 | WP1 freeze: reproducibility pass and deliverable report | pending |

## 8. Deliverables and Exit Criteria

WP1 is complete when all conditions are met:

- one-command run for each core experiment (`E1` through `E5`),
- repeatable metrics (<=5% variance on repeated short runs for latency),
- aggregated and disaggregated comparison report generated,
- outputs versioned by `run_id` with full config + hardware metadata.

## 9. Risks Specific to Low-Budget Hardware

| Risk | Impact | Mitigation |
| :---- | :---- | :---- |
| OOM at long prompts | blocks E1/E5 high-length runs | use smaller batch, quantization, reduced max length tier |
| benchmark runtime too long | delayed iteration | fixed-size benchmark subsets for WP1 |
| unstable latency noise | weak conclusions | warmup standardization + repeated trials + p50/p95 reporting |
| single-GPU OOM in real disaggregated 8B mode | blocks E3 runs on single-device hosts | execute disaggregated real runs on multi-GPU, use explicit GPU assignment, keep single-GPU disaggregated in simulated mode |
| disaggregated prototype complexity | schedule slip | stage as simple broker first, optimize in WP2 |

## 10. Decision Log (Locked for WP1)

1. **Model:** `Qwen/Qwen3-8B` is the primary baseline model.
2. **Serving backend:** `vLLM` only in WP1 for execution speed and lower integration risk.
3. **Hardware profile:** single-GPU baseline for aggregated runs, with multi-GPU planned for real disaggregated 8B runs.
4. **Benchmark depth:** fixed-size subsets for LongBench/RULER/InfinityBench in WP1.
5. **Disaggregation schedule:** disaggregated prototype is implemented; real 8B disaggregated experiments should run with multi-GPU placement.
6. **Training strategy:** pretrained/frozen-base approach for WP1-WP2; no full-model pretraining.
7. **Context policy:** run native 32K first, then YaRN stress subset up to 131K where feasible.
8. **Thinking mode policy:** `enable_thinking=False` for primary systems benchmarks; optional separate analysis later.

## 11. Near-Term Implications (Post T10)

- `E1/E2` are now execution tasks rather than instrumentation tasks; collectors and report paths are already in place.
- `E3` now has transfer/stall metrics and GPU assignment metadata available in run artifacts.
- Multi-GPU disaggregated execution is now the operational default for real 8B comparisons; single-GPU disaggregated should remain simulated/debug.
- `T11-T16` are the primary remaining blockers for WP1 completion (`G2` and `G3` closure depends on experiment execution and comparison harnesses).
