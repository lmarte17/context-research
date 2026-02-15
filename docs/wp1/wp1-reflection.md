# WP1 Reflection: What We Built, Why We Built It, and What It Enables

Prepared: February 13, 2026  
Scope: Work Package 1 (Baselines and Instrumentation)  
Status: Complete (`T1-T16`, `E0-E6`, `G1-G4`)

## 1. Why WP1 Existed

WP1 was designed to remove ambiguity before algorithmic work begins. The project thesis depends on proving improvements under hard constraints (memory, latency, and SLO goodput), so WP1 focused on building a trustworthy baseline and measurement substrate first.

The core intent was:

- make experiments reproducible and auditable,
- separate "systems improvement" from "model-change improvement",
- establish both aggregated and disaggregated serving baselines,
- and produce artifacts that can be reused directly in WP2-WP4.

Without this foundation, later gains from routing, compression, or retrieval leaves would be hard to attribute and easy to overclaim.

## 2. What WP1 Delivered

WP1 closed all planned implementation and experiment gates:

- `T1-T16`: scaffold, serving/scheduling layers, collectors, benchmark harness, and E3-E6 runners.
- `E0-E6`: smoke/reproducibility, latency sweep, concurrency sweep, mode comparison, capability subset, memory decomposition, and YaRN stress path.
- `G1-G4`: all milestone gates satisfied with runnable one-command workflows.

Concrete deliverable quality at WP1 close:

- one-command experiment/report paths for all WP1 tracks,
- run-level metadata and config hashing for traceability,
- standardized output artifacts and markdown reporting,
- disaggregated transfer accounting (bytes/time/throughput/stall ratio),
- deterministic benchmark subsets for repeatable quality checks.

## 3. Architectural Decisions and Rationale

### 3.1 Decision Summary

| Decision | Why we made it | Consequence |
| :---- | :---- | :---- |
| Use `vLLM` as the only WP1 backend | Minimize integration variance and move fast on baseline credibility | Faster WP1 closure; deferred multi-backend complexity to WP2+ |
| Enforce strict real-backend mode by default | Prevent false-positive "green" runs from accidental simulation | Higher confidence in baseline evidence; explicit opt-in needed for simulated debug |
| Build aggregated first, then disaggregated | Get a stable control baseline before adding transfer/scheduling complexity | Cleaner deltas in E3 and lower early debugging risk |
| Keep model policy pretrained/frozen (`Qwen/Qwen3-8B`) | WP1 research object is systems behavior, not finetuning | Causal attribution improves; router/adaptation learning postponed |
| Lock to budget-first hardware profile | Keep execution realistic for constrained environments | Needed careful scaling strategy (32K first, then YaRN stress path) |
| Add deterministic subset loader for quality benchmarks | Ensure repeatable capability comparisons with bounded runtime | Reliable E4/E6 comparisons; easier CI-style regression checks |
| Instrument KV transfer in disaggregated path | Disaggregation claims depend on transfer overhead visibility | E3 has meaningful overhead accounting instead of latency-only comparisons |
| Standardize run metadata + report generation | Treat each run as an auditable evidence packet | Reduced manual analysis and better handoff across contributors |

### 3.2 Why This Matters Architecturally

WP1 intentionally treated baseline engineering as part of the research contribution. The architecture now enforces:

- strict execution semantics (real vs simulated is explicit),
- measurement primitives aligned with project success criteria (TTFT/TPOT/goodput/memory/transfer),
- and stable experiment surfaces (`E0-E6`) that later WPs can extend rather than rewrite.

This lowers the risk of "benchmark drift" when WP2 introduces new memory leaves and WP3 introduces learned routing.

## 4. Key Tradeoffs and Discussion Points

### 4.1 Single Backend vs Generality

Tradeoff:

- We optimized for one reliable backend (`vLLM`) instead of backend plurality.

Why acceptable in WP1:

- The immediate bottleneck was evidence quality and execution throughput.
- Backend diversity without stable instrumentation would have increased noise.

What to watch next:

- WP2 should preserve the existing interfaces so multi-backend expansion does not fork experiment semantics.

### 4.2 Real-Only Default vs Developer Convenience

Tradeoff:

- Strict real backend can make local iteration harder when GPU setup is fragile.

Why acceptable in WP1:

- Baseline trust is more important than convenience in a measurement phase.
- Simulation remains available for controlled debug via explicit flagging.

What to watch next:

- Keep simulated mode available for fast pipeline checks, but never mix it with publishable evidence.

### 4.3 Disaggregated Ambition vs Hardware Reality

Tradeoff:

- Real disaggregated 8B runs generally require multi-GPU placement; single-GPU hosts often OOM.

Why acceptable in WP1:

- The architecture and accounting paths were implemented and validated.
- Operational policy is explicit: real disaggregated on multi-GPU, simulated/debug on single-GPU.

What to watch next:

- Final comparison packets should prioritize multi-GPU real disaggregated evidence.

### 4.4 Long Context Coverage vs Stability

Tradeoff:

- Extending beyond native 32K required YaRN stress configuration and host compatibility fixes.

Why acceptable in WP1:

- The project needs a path toward 131K stress behavior.
- Rope schema compatibility (`type`/`rope_type`) was necessary for portability.

What to watch next:

- Keep long-context config handling explicit and versioned; avoid hidden runtime patching.

## 5. What We Intentionally Deferred to WP2+

Deferrals were deliberate, not omissions:

- multi-backend serving abstraction beyond `vLLM`,
- advanced disaggregation optimization (beyond minimal broker),
- router training or finetuning,
- multi-GPU tier placement studies.

This sequencing preserved focus: WP1 built reliable baselines first, then left optimization and policy learning for the correct phase.

## 6. How WP1 Changes the Starting Point for Later WPs

### 6.1 WP2 (Leaf Modules)

WP2 inherits:

- stable serving/scheduler interfaces,
- reusable profiling collectors,
- deterministic quality and systems evaluation tracks,
- existing comparison structure (`E3`, `E5`) for measuring leaf-level impact.

Implication:

- WP2 can focus on leaf behavior (L1-L5, optional L0 integration) rather than rebuilding instrumentation.

### 6.2 WP3 (Router + Scheduler Learning)

WP3 inherits:

- hard-budget measurement hooks already present in artifacts,
- disaggregated transfer metrics needed for cost-aware routing objectives,
- explicit aggregated/disaggregated baselines for fair policy evaluation.

Implication:

- Learned policies can be judged against concrete SLO and transfer-aware constraints, not proxy metrics.

### 6.3 WP4-WP5 (End-to-End and Scaling)

WP4/WP5 inherit:

- runnable end-to-end experiment workflows,
- existing goodput and latency reporting shape,
- GPU assignment and transfer accounting patterns for multi-device scaling analysis.

Implication:

- Later integration work can concentrate on control-plane composition and tier cost models.

## 7. Residual Risks Carried Forward

The most important residual risks after WP1 are:

- evidence quality split between strict-real and simulated validation runs in some tracks,
- transfer overhead sensitivity in disaggregated setups under contention,
- potential integration overhead from future control logic on critical decode paths,
- and hardware-specific variance (GPU SKU and host runtime differences).

Current mitigation posture:

- annotate hardware and mode in every report,
- keep strict-real runs as reference for final comparisons,
- preserve lightweight control interfaces to limit runtime overhead,
- and continue one-command reproducibility discipline for each experiment track.

## 8. Lessons for Future Contributors

- Baseline credibility is an architectural feature, not a reporting afterthought.
- "Complete" means runnable end-to-end with artifacts, not just implemented classes.
- Explicit mode boundaries (real vs simulated) prevent accidental evidence contamination.
- Deterministic subsets are essential for fast, reliable long-context capability iteration.
- Disaggregation claims are only as strong as transfer accounting.

## 9. Suggested Questions for WP2 Kickoff

- Which leaf module gives the best quality-per-memory gain on `E4` and `E5` first?
- How should leaf insertion preserve `E3` comparability without changing baseline semantics?
- What is the minimum additional control overhead acceptable on decode paths?
- Which WP1 reports should be elevated into a fixed "evidence packet" template for WP2 reviews?

## 10. Source Documents Used

- `docs/research-project-plan.docx.md`
- `docs/wp1/wp1-implementation-roadmap.md`
- `docs/wp1/wp1-task-backlog.md`
- `docs/wp1/wp1-progress-log.md`
