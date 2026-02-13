from __future__ import annotations

from context_research.benchmarks.base import BenchmarkSample


def default_samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-001",
            prompt=(
                "In the old manor, Eleanor kept a ledger where she wrote one line every dawn. "
                "When the storm arrived and the windows shook all night, she still climbed to the attic, "
                "lit a candle, and wrote the single word that mattered most to her family."
            ),
            reference="endure",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-002",
            prompt=(
                "The captain reread the map until the ink blurred. At sunrise the crew asked for direction. "
                "He tapped the coast one final time and spoke the harbor name they had chased for months."
            ),
            reference="meridian",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-003",
            prompt=(
                "Marta sorted the letters by year and tied each bundle with blue thread. "
                "At the bottom of the trunk she found a postcard with no stamp and one sentence on the back, "
                "ending with the city where they first met."
            ),
            reference="lisbon",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-004",
            prompt=(
                "The apprentice copied formulas until midnight, but the final line in the notebook had been erased. "
                "On the bench sat three vials and a note: mix only the one labeled with the color of winter sky."
            ),
            reference="azure",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-005",
            prompt=(
                "After ten years abroad, Jonah returned to the village market. "
                "The baker recognized him instantly and handed him the same pastry he bought as a child, "
                "filled with tart fruit from the northern orchards."
            ),
            reference="plum",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-006",
            prompt=(
                "The orchestra paused before the encore. The conductor lowered his baton and asked for silence. "
                "Then he requested the movement named after the evening star visible above the river."
            ),
            reference="venus",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-007",
            prompt=(
                "Ravi repaired clocks in a narrow shop with no sign. "
                "When travelers asked for the secret of his precision, he pointed to a brass instrument "
                "hanging by the door and said he trusted only one measure."
            ),
            reference="seconds",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-008",
            prompt=(
                "The botanist cataloged every fern in the greenhouse. "
                "The rarest species opened once a year at dusk, revealing petals shaped like a tiny lantern."
            ),
            reference="lantern",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-009",
            prompt=(
                "The engineer warned that the bridge could hold a hundred carts if they crossed in rhythm. "
                "Drummers set the pace, and the first beat echoed over the water as the convoy began."
            ),
            reference="rhythm",
            metadata={"domain": "long_narrative"},
        ),
        BenchmarkSample(
            task="pg19_subset",
            sample_id="pg19-010",
            prompt=(
                "At the museum archive, Delia restored a torn map with rice paste and patience. "
                "Near the legend she uncovered a faded symbol indicating a hidden well beneath the old square."
            ),
            reference="well",
            metadata={"domain": "long_narrative"},
        ),
    ]

