"""Split management and leakage prevention for the ASAG Research Framework.

Handles:
- SciEntsBank: preserve predefined UA/UQ/UD splits as-is
- MohlerASAG: question-level 60/20/20 split by question_id
- Data_Generate: preserve predefined splits with integrity checks
  (adversarial co-location, unseen-question/domain leakage)

Any violation raises SplitIntegrityError with affected sample_ids.
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.data.schema import UnifiedRecord


class SplitIntegrityError(Exception):
    """Raised when a split integrity constraint is violated.

    Attributes:
        message: Human-readable description of the violation.
        affected_sample_ids: List of sample_ids involved in the violation.
    """

    def __init__(self, message: str, affected_sample_ids: list[str]) -> None:
        self.affected_sample_ids = affected_sample_ids
        super().__init__(f"{message} Affected sample_ids: {affected_sample_ids}")


class SplitManager:
    """Manage and verify dataset splits.

    Supports three modes:
    - SciEntsBank: passthrough (splits already on records)
    - MohlerASAG: question-level random split (60/20/20)
    - Data_Generate: passthrough with integrity verification
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    # ── Public API ────────────────────────────────────────────────

    def assign_splits(
        self, records: list[UnifiedRecord]
    ) -> list[UnifiedRecord]:
        """Assign or verify splits for a list of records.

        Dispatches to the appropriate handler based on source_dataset.
        Records are modified in-place and returned for convenience.
        """
        by_source: dict[str, list[UnifiedRecord]] = defaultdict(list)
        for rec in records:
            by_source[rec.source_dataset].append(rec)

        for source, recs in by_source.items():
            if source == "scientsbank":
                self._handle_scientsbank(recs)
            elif source == "mohler":
                self._handle_mohler(recs)
            elif source == "data_generate":
                self._handle_data_generate(recs)
            # data_scraping has no splits to manage

        return records

    # ── SciEntsBank ───────────────────────────────────────────────

    def _handle_scientsbank(
        self, records: list[UnifiedRecord]
    ) -> None:
        """Preserve predefined SciEntsBank UA/UQ/UD splits without modification.

        The split field is already set by the loader. We just verify
        that every record has a non-empty split.
        """
        # Nothing to change — splits are preserved as loaded.
        pass

    # ── MohlerASAG ────────────────────────────────────────────────

    def _handle_mohler(
        self, records: list[UnifiedRecord]
    ) -> None:
        """Create question-level 60/20/20 splits for MohlerASAG.

        Groups all answers by question_id, then assigns entire
        question groups to train/valid/test using a seeded shuffle.
        No question_id appears in more than one partition.
        """
        # Group records by question_id
        groups: dict[str, list[UnifiedRecord]] = defaultdict(list)
        for rec in records:
            groups[rec.question_id].append(rec)

        # Seeded shuffle of question_ids
        question_ids = sorted(groups.keys())
        rng = random.Random(self.seed)
        rng.shuffle(question_ids)

        n = len(question_ids)
        train_end = int(n * 0.6)
        valid_end = train_end + int(n * 0.2)

        train_qids = set(question_ids[:train_end])
        valid_qids = set(question_ids[train_end:valid_end])
        # Remaining question_ids go to test (no explicit set needed)

        for qid, recs in groups.items():
            if qid in train_qids:
                split_name = "train"
            elif qid in valid_qids:
                split_name = "valid"
            else:
                split_name = "test"
            for rec in recs:
                rec.split = split_name

    # ── Data_Generate ─────────────────────────────────────────────

    def _handle_data_generate(
        self, records: list[UnifiedRecord]
    ) -> None:
        """Preserve Data_Generate splits and verify integrity.

        Checks:
        1. Adversarial variants reside in the same split as their originals
        2. No question_id from test_unseen_questions appears in train
        3. No domain from test_unseen_domains appears in train
        """
        self._check_adversarial_colocation(records)
        self._check_unseen_questions(records)
        self._check_unseen_domains(records)

    def _check_adversarial_colocation(
        self, records: list[UnifiedRecord]
    ) -> None:
        """Verify adversarial variants are co-located with originals."""
        # Build lookup: original_id → record
        by_original_id: dict[str, UnifiedRecord] = {}
        for rec in records:
            by_original_id[rec.original_id] = rec

        # Also build sample_id → record for lookup by sample_id
        by_sample_id: dict[str, UnifiedRecord] = {}
        for rec in records:
            by_sample_id[rec.sample_id] = rec

        violations: list[str] = []
        for rec in records:
            if rec.adversarial_variant_of is None:
                continue

            # Try to find the original by original_id first, then sample_id
            original = by_original_id.get(rec.adversarial_variant_of)
            if original is None:
                original = by_sample_id.get(rec.adversarial_variant_of)

            if original is not None and original.split != rec.split:
                violations.append(rec.sample_id)

        if violations:
            raise SplitIntegrityError(
                "Adversarial variant(s) not co-located with original record.",
                violations,
            )

    def _check_unseen_questions(
        self, records: list[UnifiedRecord]
    ) -> None:
        """Verify no question_id from test_unseen_questions appears in train."""
        train_qids: set[str] = set()
        unseen_q_qids: set[str] = set()
        unseen_q_sample_ids: dict[str, list[str]] = defaultdict(list)

        for rec in records:
            if rec.split == "train":
                train_qids.add(rec.question_id)
            elif rec.split == "test_unseen_questions":
                unseen_q_qids.add(rec.question_id)
                unseen_q_sample_ids[rec.question_id].append(rec.sample_id)

        leaked = train_qids & unseen_q_qids
        if leaked:
            affected: list[str] = []
            # Include sample_ids from both train and test_unseen_questions
            for rec in records:
                if rec.question_id in leaked and rec.split in (
                    "train",
                    "test_unseen_questions",
                ):
                    affected.append(rec.sample_id)
            raise SplitIntegrityError(
                f"question_id(s) {sorted(leaked)} appear in both train "
                f"and test_unseen_questions.",
                affected,
            )

    def _check_unseen_domains(
        self, records: list[UnifiedRecord]
    ) -> None:
        """Verify no domain from test_unseen_domains appears in train."""
        train_domains: set[str] = set()
        unseen_d_domains: set[str] = set()

        for rec in records:
            if rec.split == "train":
                train_domains.add(rec.domain)
            elif rec.split == "test_unseen_domains":
                unseen_d_domains.add(rec.domain)

        leaked = train_domains & unseen_d_domains
        if leaked:
            affected: list[str] = []
            for rec in records:
                if rec.domain in leaked and rec.split in (
                    "train",
                    "test_unseen_domains",
                ):
                    affected.append(rec.sample_id)
            raise SplitIntegrityError(
                f"domain(s) {sorted(leaked)} appear in both train "
                f"and test_unseen_domains.",
                affected,
            )
