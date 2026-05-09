"""Data quality audit tooling for the ASAG Research Framework.

Provides a ``DataAuditor`` class that operates on lists of
:class:`~src.data.schema.UnifiedRecord` and surfaces data-quality
issues described in Requirement 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.data.schema import UnifiedRecord

# ── Constants ─────────────────────────────────────────────────────────

_LOW_CONFIDENCE_THRESHOLD = 0.85

_CALCULATION_KEYWORDS = re.compile(
    r"\b(?:calculate|compute|how many|what is the value|convert|"
    r"determine the|find the value|evaluate|solve)\b",
    re.IGNORECASE,
)

_UNIT_WORDS = re.compile(
    r"\b(?:meters?|kilometres?|kilometers?|kg|kilograms?|grams?|"
    r"litres?|liters?|miles?|feet|foot|inches?|centimeters?|"
    r"centimetres?|millimeters?|millimetres?|seconds?|minutes?|"
    r"hours?|joules?|watts?|newtons?|volts?|amperes?|amps?|"
    r"celsius|fahrenheit|kelvin|mol|moles?|pounds?|ounces?|"
    r"gallons?|mph|km/h|m/s)\b",
    re.IGNORECASE,
)

_NUMERIC_EXPRESSION = re.compile(r"\d+\.?\d*\s*[\+\-\*/÷×=]\s*\d+")

_DIGITS = re.compile(r"\d")


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class LabelDistribution:
    """Label counts and percentages for a single source dataset."""

    source_dataset: str
    label_5way: dict[str, int] = field(default_factory=dict)
    label_3way: dict[str, int] = field(default_factory=dict)
    label_2way: dict[str, int] = field(default_factory=dict)

    @property
    def label_5way_pct(self) -> dict[str, float]:
        total = sum(self.label_5way.values())
        if total == 0:
            return {}
        return {k: v / total * 100 for k, v in self.label_5way.items()}

    @property
    def label_3way_pct(self) -> dict[str, float]:
        total = sum(self.label_3way.values())
        if total == 0:
            return {}
        return {k: v / total * 100 for k, v in self.label_3way.items()}

    @property
    def label_2way_pct(self) -> dict[str, float]:
        total = sum(self.label_2way.values())
        if total == 0:
            return {}
        return {k: v / total * 100 for k, v in self.label_2way.items()}


@dataclass
class AuditReport:
    """Aggregated audit results."""

    label_distributions: list[LabelDistribution] = field(default_factory=list)
    low_confidence_records: list[UnifiedRecord] = field(default_factory=list)
    not_found_reference_records: list[UnifiedRecord] = field(default_factory=list)
    short_answer_records: list[UnifiedRecord] = field(default_factory=list)
    numerical_question_count: int = 0
    conceptual_question_count: int = 0


# ── Core auditor ──────────────────────────────────────────────────────


class DataAuditor:
    """Runs data-quality checks on a collection of ``UnifiedRecord`` objects."""

    # ── 6.1  Label distribution ───────────────────────────────────────

    @staticmethod
    def label_distributions(
        records: Sequence[UnifiedRecord],
    ) -> list[LabelDistribution]:
        """Compute label counts and percentages per source dataset."""
        by_source: dict[str, list[UnifiedRecord]] = {}
        for rec in records:
            by_source.setdefault(rec.source_dataset, []).append(rec)

        results: list[LabelDistribution] = []
        for source, recs in sorted(by_source.items()):
            dist = LabelDistribution(source_dataset=source)
            for rec in recs:
                if rec.label_5way is not None:
                    dist.label_5way[rec.label_5way] = (
                        dist.label_5way.get(rec.label_5way, 0) + 1
                    )
                if rec.label_3way is not None:
                    dist.label_3way[rec.label_3way] = (
                        dist.label_3way.get(rec.label_3way, 0) + 1
                    )
                if rec.label_2way is not None:
                    dist.label_2way[rec.label_2way] = (
                        dist.label_2way.get(rec.label_2way, 0) + 1
                    )
            results.append(dist)
        return results

    # ── 6.2  Low-confidence records ───────────────────────────────────

    @staticmethod
    def low_confidence_records(
        records: Sequence[UnifiedRecord],
        threshold: float = _LOW_CONFIDENCE_THRESHOLD,
    ) -> list[UnifiedRecord]:
        """Return records with ``annotation_confidence`` below *threshold*."""
        return [
            rec
            for rec in records
            if rec.annotation_confidence is not None
            and rec.annotation_confidence < threshold
        ]

    # ── 6.3  "Not found" reference answers ────────────────────────────

    @staticmethod
    def not_found_references(
        records: Sequence[UnifiedRecord],
    ) -> list[UnifiedRecord]:
        """Return Data_Scraping records with reference_answer "Not found"."""
        return [
            rec
            for rec in records
            if rec.source_dataset == "data_scraping"
            and rec.reference_answer == "Not found"
        ]

    # ── 6.4  Short student answers ────────────────────────────────────

    @staticmethod
    def short_student_answers(
        records: Sequence[UnifiedRecord],
        min_tokens: int = 3,
    ) -> list[UnifiedRecord]:
        """Return records with fewer than *min_tokens* tokens."""
        return [
            rec
            for rec in records
            if len(rec.student_answer.split()) < min_tokens
        ]

    # ── 6.5  Stratified audit sample ──────────────────────────────────

    @staticmethod
    def stratified_sample(
        records: Sequence[UnifiedRecord],
        n: int,
        seed: int = 42,
    ) -> list[UnifiedRecord]:
        """Select *n* records stratified by ``label_5way`` and ``source_dataset``.

        Uses deterministic sampling via Python's ``random`` module seeded
        with *seed*.  When a stratum has fewer records than its
        proportional share, all records from that stratum are included
        and the remainder is redistributed.
        """
        import random as _random

        rng = _random.Random(seed)

        # Group by (label_5way, source_dataset)
        strata: dict[tuple[str | None, str], list[UnifiedRecord]] = {}
        for rec in records:
            key = (rec.label_5way, rec.source_dataset)
            strata.setdefault(key, []).append(rec)

        total = len(records)
        if total == 0 or n <= 0:
            return []

        selected: list[UnifiedRecord] = []
        remaining_n = n

        # Sort strata keys for determinism
        sorted_keys = sorted(strata.keys(), key=lambda k: (str(k[0]), k[1]))

        # First pass: proportional allocation
        allocations: dict[tuple[str | None, str], int] = {}
        for key in sorted_keys:
            proportion = len(strata[key]) / total
            alloc = int(proportion * n)
            alloc = min(alloc, len(strata[key]))
            allocations[key] = alloc

        # Distribute remainder
        allocated_total = sum(allocations.values())
        remaining_n = n - allocated_total

        # Give extra slots to strata that still have capacity
        for key in sorted_keys:
            if remaining_n <= 0:
                break
            capacity = len(strata[key]) - allocations[key]
            extra = min(remaining_n, capacity)
            allocations[key] += extra
            remaining_n -= extra

        # Sample from each stratum
        for key in sorted_keys:
            pool = list(strata[key])
            rng.shuffle(pool)
            selected.extend(pool[: allocations[key]])

        return selected

    # ── 6.6  Numerical/computational question detection ───────────────

    @staticmethod
    def is_numerical_question(question: str) -> bool:
        """Heuristic: does *question* look numerical or computational?"""
        if _CALCULATION_KEYWORDS.search(question):
            return True
        if _UNIT_WORDS.search(question):
            return True
        if _NUMERIC_EXPRESSION.search(question):
            return True
        # Contains digits alongside math-related context
        if _DIGITS.search(question) and _UNIT_WORDS.search(question):
            return True
        return False

    @staticmethod
    def numerical_question_counts(
        records: Sequence[UnifiedRecord],
    ) -> tuple[int, int]:
        """Count numerical vs conceptual questions among Data_Scraping records.

        Returns ``(numerical_count, conceptual_count)``.
        """
        seen_questions: dict[str, bool] = {}
        for rec in records:
            if rec.source_dataset != "data_scraping":
                continue
            if rec.question_id in seen_questions:
                continue
            seen_questions[rec.question_id] = DataAuditor.is_numerical_question(
                rec.question
            )

        numerical = sum(1 for v in seen_questions.values() if v)
        conceptual = sum(1 for v in seen_questions.values() if not v)
        return numerical, conceptual

    # ── Full audit ────────────────────────────────────────────────────

    @staticmethod
    def full_audit(records: Sequence[UnifiedRecord]) -> AuditReport:
        """Run all audit checks and return an :class:`AuditReport`."""
        num, con = DataAuditor.numerical_question_counts(records)
        return AuditReport(
            label_distributions=DataAuditor.label_distributions(records),
            low_confidence_records=DataAuditor.low_confidence_records(records),
            not_found_reference_records=DataAuditor.not_found_references(records),
            short_answer_records=DataAuditor.short_student_answers(records),
            numerical_question_count=num,
            conceptual_question_count=con,
        )
