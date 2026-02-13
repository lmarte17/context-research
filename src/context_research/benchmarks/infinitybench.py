from __future__ import annotations

from context_research.benchmarks.base import BenchmarkSample


def default_samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-001",
            prompt=(
                "Multi-hop prompt: Team Redwood owns parser. Parser sends output to Atlas cache. "
                "Atlas cache feeds Vega ranker. Question: Which component receives parser output?"
            ),
            reference="Atlas cache",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-002",
            prompt=(
                "Multi-hop prompt: Report X was drafted by Mina, reviewed by Alex, and approved by Rowan. "
                "Question: Who approved Report X?"
            ),
            reference="Rowan",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-003",
            prompt=(
                "Multi-hop prompt: The first relay node is N1, backup node is N2, final sink is N5. "
                "Traffic falls back to backup before sink on failure. "
                "Question: Which node is used as backup?"
            ),
            reference="N2",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-004",
            prompt=(
                "Multi-hop prompt: Document A references Appendix C, which defines metric M7 as recall at 20. "
                "Question: What does M7 measure?"
            ),
            reference="recall at 20",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-005",
            prompt=(
                "Multi-hop prompt: Warehouse East ships to Hub North, then to Store 18. "
                "Warehouse West ships to Hub South, then to Store 22. "
                "Question: Which hub serves Store 18?"
            ),
            reference="Hub North",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-006",
            prompt=(
                "Multi-hop prompt: Dataset D1 trains model P. Model P initializes model Q. "
                "Model Q powers service R. Question: Which model powers service R?"
            ),
            reference="model Q",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-007",
            prompt=(
                "Multi-hop prompt: Contract renewal requires legal signoff after finance review. "
                "Finance review starts only after procurement intake. "
                "Question: What must happen immediately before legal signoff?"
            ),
            reference="finance review",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-008",
            prompt=(
                "Multi-hop prompt: API gateway logs to stream S3, stream S3 writes to table T9. "
                "Question: Which table receives gateway logs?"
            ),
            reference="T9",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-009",
            prompt=(
                "Multi-hop prompt: The migration starts in phase alpha, then phase beta, then phase gamma. "
                "Only beta touches user sessions. Question: Which phase touches user sessions?"
            ),
            reference="phase beta",
            metadata={"category": "multi_hop"},
        ),
        BenchmarkSample(
            task="infinitybench_subset",
            sample_id="infinity-010",
            prompt=(
                "Multi-hop prompt: Alert A pages on-call if error budget drops below two percent. "
                "Error budget depends on failed requests over rolling seven days. "
                "Question: Over what window is the budget computed?"
            ),
            reference="rolling seven days",
            metadata={"category": "multi_hop"},
        ),
    ]

