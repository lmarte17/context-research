**RESEARCH PROJECT PLAN**

------------------------

Leaf-Spine Multiplexed Memory for Longer Context in Large Language Models

*A Unified Control Plane for Hardware-Aware Memory Bandwidth Allocation*

Prepared: February 2026  
Status: Planning Phase (Modernized to 2024-2025 SOTA Baselines)

# **1. Executive Summary**

This project investigates whether a **unified multiplexed control plane** can coordinate heterogeneous memory leaves under fixed hardware and SLO budgets, so long-context quality improves while memory growth and latency growth remain bounded.

The updated (2026) framing treats long-context inference as a **joint systems + routing problem** with three non-optional ingredients:

1. **Disaggregated serving substrate** (prefill/decode separation and KV transfer-aware scheduling), following DistServe, Mooncake, MemServe, and P/D-Serve.
2. **Adaptive multi-leaf memory** (hot KV, sink/heavy-hitter, layered compression, semantic chunking, retrieval overlays), informed by PyramidInfer, PyramidKV, ChunkKV, KVzip, H2O, SnapKV, and KIVI.
3. **Modern evaluation for true long-context behavior** (RULER, InfinityBench, BABILong, LongGenBench), not NIAH-only.

Novelty remains the same: the contribution is the **coordinated control plane** that allocates bandwidth and memory across leaves and tiers, not any single compression or retrieval technique.

# **2. Research Question and Thesis**

## **2.1 Core Research Question**

Can we learn (or engineer) a stable spine policy that allocates fixed memory/bandwidth/latency budgets across heterogeneous leaves and tiers (hot KV, sink/heavy-hitter, layered compressed KV, semantic-chunk KV, and retrieval overlays) so effective long-context performance improves while TTFT/TPOT and peak memory scale gracefully as raw context grows?

## **2.2 Thesis Statement**

A two-level, hardware-aware control plane (cluster-level disaggregation scheduler + token/layer-level spine router) can outperform single-technique methods by achieving:

- better quality under fixed memory budgets,
- sublinear non-weight memory growth,
- flatter TPOT-vs-context slope,
- higher SLO-constrained goodput in long-context serving.

## **2.3 Novelty Claim**

Prior work optimizes isolated components. This project unifies them as a constrained allocation problem with a learnable controller that spans:

- intra-request memory routing,
- inter-request disaggregated scheduling,
- tier placement (HBM/CPU/SSD),
- and uncertainty-triggered retrieval/compression usage.

# **3. Architecture Overview**

## **3.1 Two-Level Control Plane (Updated)**

The modernized architecture has two coordinated controllers:

- **Global scheduler (cluster level):** chooses prefill/decode placement, P/D ratio, and KV transfer path.
  - Inspired by DistServe, Mooncake, MemServe, P/D-Serve, LoongServe.
- **Local spine router (request/layer/token level):** allocates attention and page budgets across leaves.
  - Inspired by Routing Transformer and Switch-style stable gating.

This replaces the older single-router assumption and aligns with current production-scale long-context serving.

## **3.2 Leaf Types (Memory Streams)**

| Leaf | Description | Storage / Precision | Prior Work Basis |
| :---- | :---- | :---- | :---- |
| **L0: Prefill Sparse Compute** | Dynamic sparse prefill path for very long prompts | GPU kernel path | MInference 1.0 |
| **L1: Hot Exact KV** | Recent-window exact KV with IO-aware kernels | FP16/BF16 in HBM | FlashAttention-3, PagedAttention |
| **L2: Anchors + Heavy Hitters** | Persistent sink tokens and high-utility tokens | FP16 in HBM | StreamingLLM, H2O, SnapKV |
| **L3: Layer-Adaptive Compressed KV** | Layer-wise variable KV budget + quantized cold KV | INT4/INT2 + mixed precision | PyramidInfer, PyramidKV, KIVI |
| **L4: Semantic-Chunk KV** | Chunk-level semantic retention to reduce token-fragmentation errors | Mixed precision + shared indices | ChunkKV, KVzip |
| **L5: Retrieval Overlay** | On-demand external context materialized as transient attention/KV overlays | CPU/SSD index + GPU buffers | RAG, RETRO, Memorizing Transformer |

## **3.3 Multiplexing Modalities**

- **Temporal multiplexing:** expensive leaves (retrieval, deep cold memory) are activated only when uncertainty novelty or entropy triggers fire.
- **Head-group multiplexing:** dedicated head groups for exact vs compressed vs retrieval channels (compatible with MQA/GQA layouts).
- **Spatial-temporal serving multiplexing:** colocate and schedule workloads with popularity/length awareness (MuxServe-style) while respecting P/D disaggregation.

## **3.4 Paging and Tiering Substrate**

Each leaf maps to a virtual KV page space with tier-aware placement:

- hot pages pinned in HBM,
- compressed pages in HBM or CPU DRAM,
- cold spill to SSD when needed,
- retrieval pages materialized on demand.

This explicitly follows the KV-centric disaggregated cache-pool direction shown by Mooncake/MemServe.

## **3.5 SOTA Alignment Changes (What Was Updated)**

| Previous Assumption | Updated SOTA-Aligned Position |
| :---- | :---- |
| Single-node inference loop is primary | Disaggregated P/D serving is first-class baseline and deployment target |
| Uniform compression policy across layers | Layer-adaptive compression is default (PyramidInfer/PyramidKV family) |
| Token-level eviction is sufficient | Semantic-chunk-aware retention is required to avoid context fragmentation |
| Throughput-only optimization focus | TTFT/TPOT/SLO goodput and KV transfer efficiency are co-primary metrics |
| NIAH-like evaluation mostly sufficient | RULER, InfinityBench, BABILong, LongGenBench required for credible claims |

# **4. Literature Map (Modernized)**

| Category | Key Works | Project Role | Key Insight Used |
| :---- | :---- | :---- | :---- |
| **IO-Aware Kernels** | FlashAttention-3 | L1 substrate | Attention remains IO-bound; Hopper-aware optimization matters |
| **Prefill Acceleration** | MInference 1.0 | L0 leaf | Dynamic sparse prefill can cut long-prompt prefill latency |
| **Disaggregated Serving** | DistServe, Mooncake, MemServe, P/D-Serve | Global control plane substrate | Prefill/decode separation and KV movement dominate SLO behavior |
| **Elastic Parallelism** | LoongServe | Multi-device scaling | Elastic sequence parallelism improves long-context utilization |
| **Serving Multiplexing** | MuxServe | Cluster-level scheduling policy | Spatial-temporal multiplexing formalism for shared resources |
| **Prefix-aware Attention** | ChunkAttention | Prefix/KV sharing optimization | Shared prefixes can reduce attention/KV overhead |
| **KV Compression (Layer-aware)** | PyramidInfer, PyramidKV | L3 leaf | Useful KV budget decreases by layer; non-uniform budgets outperform uniform |
| **KV Compression (Semantic/query-agnostic)** | ChunkKV, KVzip | L4 leaf | Semantic chunking and reusable compression reduce quality loss and latency |
| **Context Extension** | LongRoPE, Infini-attention | Stress-test regime and alternative memory channel | Million-token contexts require positional + memory-mechanism adaptation |
| **Long-Context Evaluation** | RULER, InfinityBench, BABILong, LongGenBench | Evaluation framework | Need retrieval, reasoning, and generation evaluations at real long lengths |

# **5. Work Packages and Timeline**

The project remains six WPs over ~12 months, with updated technical targets.

## **WP1: Baselines and Instrumentation (Months 1-2)**

**Goal:** Establish modern serving and evaluation baselines.

- Baselines: aggregated serving + disaggregated serving (DistServe-style and P/D-Serve-style pipeline).
- Kernel baseline: FlashAttention-3; optional prefill path with MInference 1.0.
- Benchmarks: PG-19, LongBench/SCROLLS, RULER, InfinityBench, BABILong, LongGenBench.
- Metrics: peak memory decomposition, TTFT, TPOT, SLO goodput, KV transfer bandwidth, batching throughput.

**Deliverable:** Reproducible baseline suite with profiling dashboard.

## **WP2: Modern Leaf Modules (Months 2-4)**

**Goal:** Implement interchangeable leaves on a unified page API.

- L1: exact hot KV.
- L2: sink + heavy-hitter retention.
- L3: layer-adaptive compressed KV (PyramidInfer/PyramidKV-style plus quantization).
- L4: semantic-chunk compression (ChunkKV/KVzip-inspired).
- L5: retrieval overlay leaf.
- L0: sparse prefill path integration.

**Deliverable:** Modular leaf library with per-leaf quality/latency/memory curves.

## **WP3: Two-Level Routing and Training (Months 4-6)**

**Goal:** Train and validate stable routing under hard budgets.

- Local spine router variants: heuristic, MLP gating, attention-statistics-conditioned gating.
- Global scheduler policy: P/D ratio controller + queue length/prompt-length aware placement.
- Joint objective: task loss + budget violation penalty + load-balance regularizer + SLO penalty.
- Compare per-layer vs global routing and trigger-based temporal routing.

**Deliverable:** Stable router/scheduler pair with budget compliance and low overhead.

## **WP4: End-to-End Multiplexed Serving (Months 6-8)**

**Goal:** Integrate leaves and both control loops in one serving stack.

- Integrate local leaf routing into decode loop.
- Integrate global P/D disaggregation and KV transfer path selection.
- Implement trigger-based retrieval and deep-cold activation.
- Validate bounded memory and stable latency from 4K to 128K+ contexts.

**Deliverable:** End-to-end prototype with disaggregated multiplexed serving.

## **WP5: Multi-GPU / Multi-Tier Scaling (Months 8-10)**

**Goal:** Scale across devices and memory tiers.

- Elastic sequence parallel extension (LoongServe-style).
- Tier placement study: HBM vs DRAM vs SSD cache pools.
- Router cost model includes interconnect and transfer congestion.

**Deliverable:** Scaling curves and communication-overhead break-even analysis.

## **WP6: Final Evaluation, Ablations, Paper (Months 10-12)**

**Goal:** Publication-grade evidence and open-source release.

- Full benchmark suite run.
- Core ablations from Section 6.
- Paper + artifact package + reproducibility checklist.

**Deliverable:** Submission-ready paper and release candidate repository.

# **6. Evaluation Framework**

## **6.1 Capability Metrics**

| Metric | Benchmark | What It Tests |
| :---- | :---- | :---- |
| Long-range perplexity | PG-19 | Language modeling over long dependencies |
| Long-context understanding | LongBench, SCROLLS | Document-level retrieval and reasoning |
| Synthetic long-context stress | RULER, InfinityBench | Real usable context size beyond NIAH |
| Reasoning-in-haystack | BABILong | Multi-fact reasoning across long noisy contexts |
| Long-form generation | LongGenBench | Coherent long-answer generation under long context |

## **6.2 Serving and Cost Metrics**

| Metric | Measurement | Target |
| :---- | :---- | :---- |
| Peak non-weight memory | Hot KV + compressed KV + buffers + paging overhead | Sublinear trend vs raw context |
| TTFT | Prefill latency P50/P95 | Reduced vs aggregated baseline |
| TPOT | Decode latency P50/P95 | Flatter slope vs prompt length |
| SLO goodput | Requests meeting TTFT + TPOT SLO simultaneously | Higher than disaggregated and aggregated baselines |
| KV transfer efficiency | D2D/host transfer throughput and stall ratio | Bounded overhead under load |
| Throughput under batching | tokens/s at fixed hardware | Higher than full-attention baseline |

## **6.3 Core Ablations**

| Ablation | Comparison | Tests |
| :---- | :---- | :---- |
| Routing vs static | Learned/adaptive vs fixed leaf budgets | Value of local adaptive routing |
| One-level vs two-level control | Local router only vs local + global scheduler | Value of joint serving-memory control |
| Aggregated vs P/D disaggregated | Single pool vs separated prefill/decode | Effect of substrate modernization |
| Layer-adaptive vs uniform compression | Pyramid-style vs fixed-ratio compression | Importance of layer awareness |
| Semantic chunk vs token-only compression | ChunkKV/KVzip-style vs token pruning | Semantic preservation benefit |
| Single-device vs distributed tiering | One GPU vs multi-tier/multi-device | Communication and tiering break-even |

# **7. Hardware Constraints and Systems Design**

## **7.1 Memory Hierarchy Targeting**

- Keep L1/L2 latency-critical leaves in HBM.
- Place L3/L4 in mixed HBM + DRAM with compression.
- Materialize L5 retrieval overlays transiently.
- Use FlashAttention-3-compatible layouts for hot-path compute locality.

## **7.2 Interconnect and Transfer Awareness**

Disaggregated serving performance depends on KV transfer path quality. The cost model must include:

- interconnect bandwidth and contention,
- transfer setup overhead,
- overlap potential with decode compute,
- and queueing effects under bursty traffic.

## **7.3 Budget Formalism (Revised)**

Let:

- `B_hbm`, `B_dram`, `B_ssd` = tiered memory budgets,
- `L_ttft`, `L_tpot` = latency ceilings,
- `G_slo` = minimum goodput target.

Router/scheduler outputs must satisfy all hard/soft constraints jointly:

- memory allocations within tier budgets,
- expected TTFT/TPOT within SLO,
- and goodput above `G_slo`.

# **8. Risks and Mitigations**

| Risk | Description | Impact | Mitigation |
| :---- | :---- | :---- | :---- |
| **Control instability** | Router/scheduler oscillation across leaves or P/D ratios | High | Entropy/load-balance regularization, hysteresis, safe fallback policy |
| **KV transfer bottleneck** | Disaggregation gains erased by transfer overhead | High | Transfer-path-aware scheduling, overlap compute+transfer, admission control |
| **Compression quality cliffs** | Aggressive compression breaks reasoning/generation | Medium | Layer-aware budgets, semantic chunking, dynamic rollback triggers |
| **Integration overhead** | Control plane overhead offsets gains | Medium | Keep control model lightweight, profile on critical path, cache routing decisions |
| **Benchmark mismatch** | Gains appear only on retrieval benchmarks | Medium | Include generation + reasoning benchmarks (LongGenBench, BABILong) |

# **9. Success Criteria**

Project succeeds if all conditions hold:

1. **Quality robustness:** Under equal memory budget, multiplexed system matches or exceeds best single-technique baseline on LongBench/SCROLLS and does not collapse on RULER/BABILong/LongGenBench.
2. **Memory scaling:** Peak non-weight memory grows sublinearly from 4K to 128K+ contexts.
3. **Latency scaling:** TPOT-vs-prompt-length slope is flatter than aggregated and single-technique baselines.
4. **Serving utility:** SLO-constrained goodput improves over both aggregated and naive disaggregated baselines.

# **10. Open Questions for Further Investigation**

- Should local routing operate per token, per chunk, or per layer by default?
- How should uncertainty triggers be calibrated across tasks to avoid unnecessary retrieval/cold-leaf activation?
- Can one control policy generalize across domains, or do we need task-conditioned routing heads?
- How should context-extension methods (LongRoPE/Infini-attention) be combined with KV multiplexing in a stable way?
- At what scale does joint optimization of scheduler + router become necessary versus modular heuristics?

# **11. Key References (Core / Must Cite)**

## **11.1 Serving and Systems Substrate**

- DistServe (2024): [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
- Mooncake (2024-2025): [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- MemServe (2024): [arXiv:2406.17565](https://arxiv.org/abs/2406.17565)
- P/D-Serve (2024): [arXiv:2408.08147](https://arxiv.org/abs/2408.08147)
- LoongServe (2024): [arXiv:2404.09526](https://arxiv.org/abs/2404.09526)
- MuxServe (2024): [arXiv:2404.02015](https://arxiv.org/abs/2404.02015)
- ChunkAttention (2024): [arXiv:2402.15220](https://arxiv.org/abs/2402.15220)

## **11.2 Kernels and Context Extension**

- FlashAttention-3 (2024): [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- MInference 1.0 (2024): [arXiv:2407.02490](https://arxiv.org/abs/2407.02490)
- LongRoPE (2024): [arXiv:2402.13753](https://arxiv.org/abs/2402.13753)
- Infini-attention (2024): [arXiv:2404.07143](https://arxiv.org/abs/2404.07143)

## **11.3 KV Compression and Retention**

- PyramidInfer (ACL Findings 2024): [ACL Anthology 2024.findings-acl.195](https://aclanthology.org/2024.findings-acl.195/)
- PyramidKV (2024/2025): [arXiv:2406.02069](https://arxiv.org/abs/2406.02069)
- ChunkKV (2025): [arXiv:2502.00299](https://arxiv.org/abs/2502.00299)
- KVzip (2025): [arXiv:2505.23416](https://arxiv.org/abs/2505.23416)

## **11.4 Long-Context Evaluation**

- RULER (2024): [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)
- InfinityBench (2024): [arXiv:2402.13718](https://arxiv.org/abs/2402.13718)
- BABILong (2024): [arXiv:2406.10149](https://arxiv.org/abs/2406.10149)
- LongGenBench (2024): [arXiv:2410.04199](https://arxiv.org/abs/2410.04199)

# **12. Optional References (Context and Extensions)**

These are useful for grounding, ablations, and historical comparisons, but not mandatory in the first paper draft.

- PagedAttention / vLLM (2023): [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- KIVI (2024): [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- H2O (2023): [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- SnapKV (2024): [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- StreamingLLM (2023): [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- DynamicKV (2024/2025): [arXiv:2412.12922](https://arxiv.org/abs/2412.12922)
- Gist Tokens (2023): [arXiv:2304.08467](https://arxiv.org/abs/2304.08467)
- RAG (2020): [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- RETRO (2022): [arXiv:2112.04426](https://arxiv.org/abs/2112.04426)
- Memorizing Transformer (2022): [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)
- Longformer (2020): [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- BigBird (2020): [arXiv:2007.14062](https://arxiv.org/abs/2007.14062)
- Performer (2021): [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)
- Routing Transformer (2021): [arXiv:2003.05997](https://arxiv.org/abs/2003.05997)
- Switch Transformer (2022): [JMLR page](https://jmlr.org/papers/v23/21-0998.html)
