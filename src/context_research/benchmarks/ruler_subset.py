from __future__ import annotations

from context_research.benchmarks.base import BenchmarkSample


def default_samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-001",
            prompt=(
                "Needle task: In a long sequence of random city names, the hidden key is 'quartz-914'. "
                "Question: Return only the hidden key."
            ),
            reference="quartz-914",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-002",
            prompt=(
                "Needle task: Ignore all distractors. The one required token is 'amber-771'. "
                "Question: What token must be returned?"
            ),
            reference="amber-771",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-003",
            prompt=(
                "Needle task: Context contains many numbers. The target marker is 'level-2026'. "
                "Output exactly the marker."
            ),
            reference="level-2026",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-004",
            prompt=(
                "Needle task: The correct phrase appears once: 'delta-signal-48'. "
                "Respond with that exact phrase."
            ),
            reference="delta-signal-48",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-005",
            prompt=(
                "Needle task: Hidden code in long context is 'jade-5501'. "
                "Return only the code."
            ),
            reference="jade-5501",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-006",
            prompt=(
                "Needle task: The anchor string is 'ember-204'. "
                "Question: What is the anchor string?"
            ),
            reference="ember-204",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-007",
            prompt=(
                "Needle task: Keep the response short. The required key is 'northwind-66'. "
                "Return the key."
            ),
            reference="northwind-66",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-008",
            prompt=(
                "Needle task: In this long context, the only useful token is 'glyph-311'. "
                "What is the useful token?"
            ),
            reference="glyph-311",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-009",
            prompt=(
                "Needle task: Retrieve the exact sequence 'orbit-9090'. "
                "Output just the sequence."
            ),
            reference="orbit-9090",
            metadata={"category": "needle"},
        ),
        BenchmarkSample(
            task="ruler_subset",
            sample_id="ruler-010",
            prompt=(
                "Needle task: The hidden answer is 'violet-413'. "
                "Provide the hidden answer exactly."
            ),
            reference="violet-413",
            metadata={"category": "needle"},
        ),
    ]

