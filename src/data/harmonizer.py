"""Label harmonization for the ASAG Research Framework.

Converts heterogeneous label spaces across datasets into the
unified 2-way, 3-way, and normalized score representations.
"""

from __future__ import annotations

import logging

from src.data.schema import UnifiedRecord

logger = logging.getLogger(__name__)

# ── SciEntsBank 5-way → 3-way mapping ────────────────────────────────

SCIENTSBANK_5WAY_TO_3WAY: dict[str, str] = {
    "correct": "correct",
    "partially_correct_incomplete": "partially_correct",
    "contradictory": "incorrect",
    "irrelevant": "incorrect",
    "non_domain": "incorrect",
}

# ── SciEntsBank 3-way → 2-way mapping ────────────────────────────────

LABEL_3WAY_TO_2WAY: dict[str, str] = {
    "correct": "correct",
    "partially_correct": "incorrect",
    "incorrect": "incorrect",
}


class LabelHarmonizer:
    """Harmonize labels across all source datasets.

    Supports:
    - MohlerASAG: score_raw → label_2way, label_3way, score_normalized
    - SciEntsBank: label_5way → label_3way → label_2way
    - Data_Generate: label_3way="contradictory" → "incorrect"
    - Consistency check: warn when score_normalized > 0.8
      but label_2way = "incorrect"
    """

    def __init__(self, threshold_2way: float = 2.5) -> None:
        self.threshold_2way = threshold_2way

    # ── Public API ────────────────────────────────────────────────

    def harmonize(
        self, record: UnifiedRecord
    ) -> UnifiedRecord:
        """Harmonize labels on a single record (in-place).

        Dispatches to the appropriate source-specific handler,
        then runs the consistency check.

        Returns the same record for convenience.
        """
        source = record.source_dataset

        if source == "mohler":
            self._harmonize_mohler(record)
        elif source == "scientsbank":
            self._harmonize_scientsbank(record)
        elif source == "data_generate":
            self._harmonize_data_generate(record)
        # data_scraping has no labels to harmonize

        self._consistency_check(record)
        return record

    def harmonize_all(
        self, records: list[UnifiedRecord]
    ) -> list[UnifiedRecord]:
        """Harmonize labels on a list of records (in-place)."""
        for rec in records:
            self.harmonize(rec)
        return records

    # ── MohlerASAG ────────────────────────────────────────────────

    def _harmonize_mohler(self, rec: UnifiedRecord) -> None:
        if rec.score_raw is None:
            return

        # score_normalized = score_raw / 5.0
        rec.score_normalized = rec.score_raw / 5.0

        # label_2way via configurable threshold
        if rec.score_raw >= self.threshold_2way:
            rec.label_2way = "correct"
        else:
            rec.label_2way = "incorrect"

        # label_3way via fixed boundaries
        #   [0, 1)   → incorrect
        #   [1, 4)   → partially_correct
        #   [4, 5]   → correct
        if rec.score_raw < 1.0:
            rec.label_3way = "incorrect"
        elif rec.score_raw < 4.0:
            rec.label_3way = "partially_correct"
        else:
            rec.label_3way = "correct"

    # ── SciEntsBank ───────────────────────────────────────────────

    def _harmonize_scientsbank(self, rec: UnifiedRecord) -> None:
        if rec.label_5way is None:
            return

        label_5 = rec.label_5way
        label_3 = SCIENTSBANK_5WAY_TO_3WAY.get(label_5)
        if label_3 is None:
            logger.warning(
                "Unknown SciEntsBank label_5way %r for %s",
                label_5,
                rec.sample_id,
            )
            return

        rec.label_3way = label_3
        rec.label_2way = LABEL_3WAY_TO_2WAY[label_3]

    # ── Data_Generate ─────────────────────────────────────────────

    def _harmonize_data_generate(self, rec: UnifiedRecord) -> None:
        # Remap label_3way="contradictory" → "incorrect"
        if rec.label_3way == "contradictory":
            rec.label_3way = "incorrect"

    # ── Consistency check ─────────────────────────────────────────

    def _consistency_check(self, rec: UnifiedRecord) -> None:
        if (
            rec.score_normalized is not None
            and rec.score_normalized > 0.8
            and rec.label_2way == "incorrect"
        ):
            logger.warning(
                "Inconsistency: sample_id=%s has "
                "score_normalized=%.3f but label_2way=%r",
                rec.sample_id,
                rec.score_normalized,
                rec.label_2way,
            )
