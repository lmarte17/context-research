from __future__ import annotations

from context_research.benchmarks.base import BenchmarkSample


def default_samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-001",
            prompt=(
                "Document excerpt: The city council approved the Green Transit Act in 2022, "
                "allocated $12M for electric buses, and set deployment for Q3 2024. "
                "Question: Which quarter was deployment scheduled?"
            ),
            reference="Q3 2024",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-002",
            prompt=(
                "Meeting notes: Alpha team owns ingestion, Beta team owns ranking, Gamma team owns reporting. "
                "The outage was traced to ingestion retries. Question: Which team owns the failing component?"
            ),
            reference="Alpha team",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-003",
            prompt=(
                "Article summary: The glacier has retreated 18 percent since 1990, "
                "with acceleration after 2010 due to warmer summers. "
                "Question: What percentage retreat is reported?"
            ),
            reference="18 percent",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-004",
            prompt=(
                "Policy text: Claims must be submitted within 30 days, "
                "appeals within 15 days of denial, and final review takes 10 business days. "
                "Question: How many days are allowed for appeals?"
            ),
            reference="15 days",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-005",
            prompt=(
                "Research memo: Baseline model scored 71.4 F1, retrieval-augmented scored 76.9 F1, "
                "and compressed context scored 74.2 F1. Question: Which setting had the best F1?"
            ),
            reference="retrieval-augmented",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-006",
            prompt=(
                "Project timeline: Design freeze on May 3, implementation complete on June 18, "
                "validation complete on July 2. Question: When did implementation complete?"
            ),
            reference="June 18",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-007",
            prompt=(
                "Case file: Shipment A left Denver at 09:10, reached Omaha at 14:55, "
                "and arrived in Chicago at 20:40. Question: Which city was the second stop?"
            ),
            reference="Omaha",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-008",
            prompt=(
                "Release notes: Version 2.3 fixed tokenizer drift, improved cache reuse, "
                "and removed legacy sampling flag. Question: Which issue was fixed first in this sentence?"
            ),
            reference="tokenizer drift",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-009",
            prompt=(
                "Lab log: Batch 14 used catalyst C and yielded 82 percent, "
                "Batch 15 used catalyst B and yielded 77 percent. "
                "Question: Which catalyst produced the higher yield?"
            ),
            reference="catalyst C",
            metadata={"category": "qa"},
        ),
        BenchmarkSample(
            task="longbench_subset",
            sample_id="longbench-010",
            prompt=(
                "Study guide: Chapter 2 covers memory paging, Chapter 3 covers routing policy, "
                "Chapter 4 covers disaggregated scheduling. Question: Which chapter covers routing policy?"
            ),
            reference="Chapter 3",
            metadata={"category": "qa"},
        ),
    ]

