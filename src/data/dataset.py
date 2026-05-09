"""Data_Loader public API for the ASAG Research Framework.

Provides programmatic access to unified records with filtering,
cross-dataset merging, and training-batch generation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.data.schema import UnifiedRecord

# Fields that can be used as filter keys
_FILTER_FIELDS = frozenset({
    "source_dataset",
    "domain",
    "label_5way",
    "label_3way",
    "label_2way",
    "is_adversarial",
    "usable_for_grading",
})


def _apply_filters(
    records: Iterable[UnifiedRecord],
    filters: dict | None,
) -> list[UnifiedRecord]:
    """Return records matching all filter key-value pairs."""
    if not filters:
        return list(records)

    result: list[UnifiedRecord] = []
    for rec in records:
        match = True
        for key, value in filters.items():
            if key not in _FILTER_FIELDS:
                continue
            if getattr(rec, key) != value:
                match = False
                break
        if match:
            result.append(rec)
    return result


class DataLoader:
    """Public data-loading interface for all downstream consumers.

    Records can be supplied directly (e.g. in tests) or loaded from
    unified JSONL files via a future ``from_jsonl`` class method.
    """

    def __init__(self, records: list[UnifiedRecord]) -> None:
        self._records = list(records)

        # Index: (source_dataset, split) → list[UnifiedRecord]
        self._index: dict[tuple[str, str], list[UnifiedRecord]] = defaultdict(list)
        for rec in self._records:
            self._index[(rec.source_dataset, rec.split)].append(rec)

    # ── helpers ────────────────────────────────────────────────────

    def _available_splits(self, source: str) -> list[str]:
        """Return sorted list of splits available for *source*."""
        return sorted(
            {split for src, split in self._index if src == source}
        )

    def _require_split(self, source: str, split: str) -> list[UnifiedRecord]:
        """Return records for (source, split) or raise ValueError."""
        key = (source, split)
        if key not in self._index:
            available = self._available_splits(source)
            raise ValueError(
                f"Split {split!r} does not exist for source {source!r}. "
                f"Available splits: {available}"
            )
        return self._index[key]

    # ── public API ────────────────────────────────────────────────

    def get_split(
        self,
        source: str,
        split: str,
        filters: dict | None = None,
    ) -> list[UnifiedRecord]:
        """Return all records for a specified source + split.

        Args:
            source: Source dataset name (e.g. ``"scientsbank"``).
            split: Split name (e.g. ``"train"``, ``"test_ua"``).
            filters: Optional dict of field→value filters.

        Returns:
            List of matching ``UnifiedRecord`` objects.

        Raises:
            ValueError: If the requested split does not exist for the
                given source, listing available splits.
        """
        records = self._require_split(source, split)
        return _apply_filters(records, filters)

    def get_merged(
        self,
        sources: list[tuple[str, str]],
        filters: dict | None = None,
    ) -> list[UnifiedRecord]:
        """Combine multiple (source, split) pairs into a single list.

        Each record retains its original ``source_dataset`` field.

        Args:
            sources: List of ``(source, split)`` tuples.
            filters: Optional dict of field→value filters applied
                after merging.

        Returns:
            Merged list of ``UnifiedRecord`` objects.

        Raises:
            ValueError: If any requested split does not exist.
        """
        merged: list[UnifiedRecord] = []
        for source, split in sources:
            merged.extend(self._require_split(source, split))
        return _apply_filters(merged, filters)

    def get_training_batch(
        self,
        sources: list[tuple[str, str]],
        label_field: str,
        filters: dict | None = None,
    ) -> Iterable[tuple[tuple[str, str, str], str | float]]:
        """Yield ``((question, reference_answer, student_answer), label)`` tuples.

        Args:
            sources: List of ``(source, split)`` tuples to draw from.
            label_field: Attribute name on ``UnifiedRecord`` to use as
                the label (e.g. ``"label_3way"``, ``"score_normalized"``).
            filters: Optional dict of field→value filters.

        Yields:
            ``((question, reference_answer, student_answer), label_value)``
            for each matching record.

        Raises:
            ValueError: If any requested split does not exist.
        """
        records = self.get_merged(sources, filters)
        for rec in records:
            triplet = (rec.question, rec.reference_answer, rec.student_answer)
            label = getattr(rec, label_field)
            yield triplet, label
