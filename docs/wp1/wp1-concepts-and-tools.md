# WP1 Concepts and Tools: What We Learned and Why It Matters

Prepared: February 15, 2026  
Scope: Practical concepts and tooling used in WP1  
Audience: Contributors onboarding to WP2+ who need shared systems vocabulary

## 1. Purpose of This Document

This document is a practical guide to the core systems concepts used in WP1 and why they matter to the research process.  
The goal is to make later design and evaluation decisions faster, clearer, and more defensible.

## 2. Core Tooling and Concepts

### 2.1 vLLM

What it is:

- A high-throughput LLM serving engine used as WP1’s baseline backend.

Why we used it in WP1:

- It reduced integration risk and gave a stable serving substrate while we built instrumentation and experiment workflows.

Why it matters:

- If backend behavior is unstable, latency and memory conclusions are unreliable.
- A single trusted backend in WP1 made baseline comparisons cleaner before adding multi-backend complexity.

Key lesson:

- A strong baseline backend is a research accelerator, not just an implementation detail.

### 2.2 Aggregated Serving

What it is:

- Prefill and decode run in one serving pool/path.

Why we used it in WP1:

- It is the simplest operational baseline and provides a clean control condition for disaggregated comparisons.

Why it matters:

- Every improvement claim in later WPs needs a clear "before" reference.
- Aggregated mode gives that reference for TTFT/TPOT/goodput and memory behavior.

Key lesson:

- Always establish a simple baseline path before introducing scheduling and transfer complexity.

### 2.3 Disaggregated Serving (Prefill/Decode Separation)

What it is:

- Prefill and decode are split into separate pools with handoff/transfer between them.

Why we used it in WP1:

- The project thesis assumes modern long-context serving where P/D disaggregation is first-class.

Why it matters:

- It exposes transfer overheads and scheduling effects that aggregated mode hides.
- It aligns the baseline with the architecture expected in later control-plane work.

Key lesson:

- Disaggregation gains are only credible when transfer costs are measured directly.

### 2.4 KV Transfer Accounting

What it is:

- Measurement of bytes moved, transfer time, throughput, and stall ratio in disaggregated paths.

Why we used it in WP1:

- Disaggregated performance depends on movement of KV/state between serving stages.

Why it matters:

- Without transfer accounting, latency changes can be misattributed.
- It is required input for future routing/scheduling cost models.

Key lesson:

- "Faster decode" claims are incomplete unless transfer overhead is included.

### 2.5 TTFT and TPOT

What they are:

- `TTFT` (Time To First Token): prefill/initial response latency.
- `TPOT` (Time Per Output Token): sustained decode latency.

Why we used them in WP1:

- They separate user-perceived startup speed from steady-state generation efficiency.

Why they matter:

- Different architectural changes affect TTFT and TPOT differently.
- Later policy decisions need both metrics to avoid optimizing one while regressing the other.

Key lesson:

- Latency is not one number; startup and steady-state must be tracked independently.

### 2.6 SLO and SLO-Constrained Goodput

What they are:

- `SLO` (Service Level Objective): latency targets (for example TTFT/TPOT ceilings).
- `Goodput`: throughput of requests that meet the SLOs, not just raw tokens/sec.

Why we used them in WP1:

- Raw throughput can look good while user-facing latency is unacceptable.

Why they matter:

- The project’s success criteria are service-quality aware, not throughput-only.
- Goodput is a better operational signal for real deployment behavior.

Key lesson:

- Throughput without SLO compliance is not meaningful system quality.

### 2.7 Peak Memory Decomposition

What it is:

- Breaking peak memory into components (KV and non-KV contributors, by mode and prompt regime).

Why we used it in WP1:

- The thesis requires bounded/sublinear non-weight memory growth as context scales.

Why it matters:

- Memory failures (OOM, paging pressure) can invalidate experiments before quality is evaluated.
- Decomposition enables targeted optimizations in WP2 leaves and WP3 routing.

Key lesson:

- Memory profiling must be component-level, not just a single peak number.

### 2.8 Deterministic Benchmark Subsets

What they are:

- Fixed seeded sample subsets for tasks (LongBench, RULER, InfinityBench tracks in WP1 harness).

Why we used them in WP1:

- Full benchmark suites are expensive; WP1 needed quick, repeatable quality signals.

Why they matter:

- Determinism allows fair before/after comparisons across experiments and code changes.
- It reduces false conclusions caused by sample drift.

Key lesson:

- Reproducibility is a design requirement, not a post-hoc reporting step.

### 2.9 YaRN Context Extension Path

What it is:

- A configuration path for stress-testing contexts beyond native limits (up to WP1 best-effort 131K regime).

Why we used it in WP1:

- The project targets long-context behavior beyond default context windows.

Why it matters:

- It establishes an early path for extended-context stress experiments before advanced leaf/routing methods are added.
- Compatibility handling (rope schema variants) improved cross-host reliability.

Key lesson:

- Long-context testing requires explicit configuration discipline and portability checks.

### 2.10 Run Metadata and Artifact Discipline

What it is:

- Standardized run IDs, config hashes, hardware manifests, and report artifacts per experiment.

Why we used it in WP1:

- To ensure every result can be traced, compared, and reproduced.

Why it matters:

- Later WPs will generate many ablations; without run discipline, evidence quality degrades quickly.

Key lesson:

- A result without metadata is not durable research evidence.

## 3. How These Concepts Fit Together in the WP1 Process

WP1 process chain:

1. Stable serving substrate (`vLLM`) establishes execution reliability.
2. Aggregated baseline defines the control condition.
3. Disaggregated mode adds realistic serving complexity.
4. KV transfer + latency + GPU/memory collectors make overhead visible.
5. SLO-goodput framing converts raw metrics into operational relevance.
6. Deterministic benchmark subsets provide repeatable quality checks.
7. YaRN stress path extends evaluation toward target long-context regimes.
8. Run metadata/reporting turns all outputs into reusable evidence packets.

This chain is why WP1 can support WP2-WP5 without rebuilding foundation tooling.

## 4. Why This Is Important for WP2 and Beyond

What this enables immediately:

- fair leaf-module comparisons under stable metrics,
- faster detection of regressions in latency, memory, and transfer overhead,
- and stronger attribution of gains to architecture changes (not tooling drift).

What this prevents:

- overclaiming based on non-reproducible runs,
- confusing throughput gains with real service quality,
- and hidden regressions caused by environment or benchmark variance.

## 5. Common Misreads to Avoid

- "Higher throughput means better system."  
  Not if SLO compliance drops.

- "Disaggregated is always better."  
  Not if transfer stalls dominate.

- "One successful long-context run proves scalability."  
  Not without repeated measurements and memory/latency trend evidence.

- "Simulation and real backend evidence are interchangeable."  
  They are useful for different purposes and must be labeled separately.

## 6. Practical Takeaway

WP1 taught that credible long-context systems research depends on measurement architecture as much as model architecture.  
The concepts above are the operating constraints for all later work packages, and they should remain explicit in every design, experiment, and report.

## 7. Related Documents

- `docs/wp1/wp1-reflection.md`
- `docs/wp1/wp1-implementation-roadmap.md`
- `docs/wp1/wp1-task-backlog.md`
- `docs/wp1/wp1-progress-log.md`
