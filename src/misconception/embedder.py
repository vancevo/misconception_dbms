"""Embedding strategies for misconception mining.

Provides three embedding strategies for representing incorrect student answers:
  - Strategy A (answer_only): embed(student_answer)
  - Strategy B (question_answer): embed(question ⊕ student_answer)
  - Strategy C (full_triplet): embed(question ⊕ reference_answer ⊕ student_answer)

Records are filtered to include only those where
label_5way ∈ {partially_correct_incomplete, contradictory, irrelevant}.

Supports per-question, per-domain, and global granularity levels.

NOTE: sentence_transformers is loaded lazily so the module can be imported
in environments where the library is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from src.data.schema import UnifiedRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISCONCEPTION_LABELS: frozenset[str] = frozenset(
    {"partially_correct_incomplete", "contradictory", "irrelevant"}
)


class EmbeddingStrategy(str, Enum):
    """Embedding strategy identifiers."""

    ANSWER_ONLY = "answer_only"
    QUESTION_ANSWER = "question_answer"
    FULL_TRIPLET = "full_triplet"


class Granularity(str, Enum):
    """Clustering granularity levels."""

    PER_QUESTION = "per_question"
    PER_DOMAIN = "per_domain"
    GLOBAL = "global"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Container for embedding output.

    Attributes:
        embeddings: 2-D numpy array of shape (n_records, embedding_dim).
        records: The filtered records corresponding to each row.
        strategy: The embedding strategy used.
        granularity: The granularity level used.
        group_key: The group key value (question_id, domain, or "global").
    """

    embeddings: np.ndarray
    records: list[UnifiedRecord]
    strategy: EmbeddingStrategy
    granularity: Granularity
    group_key: str


# ---------------------------------------------------------------------------
# SBERT loader (lazy)
# ---------------------------------------------------------------------------

def _load_sbert(model_name: str):
    """Lazily import and return a SentenceTransformer instance."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "sentence_transformers is required for misconception embeddings. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


# ---------------------------------------------------------------------------
# Record filtering
# ---------------------------------------------------------------------------

def filter_misconception_records(
    records: Iterable[UnifiedRecord],
) -> list[UnifiedRecord]:
    """Select records where label_5way ∈ MISCONCEPTION_LABELS.

    Only records whose ``label_5way`` is one of the three
    misconception labels are included.
    """
    return [
        r for r in records
        if r.label_5way is not None and r.label_5way in MISCONCEPTION_LABELS
    ]


# ---------------------------------------------------------------------------
# Text preparation per strategy
# ---------------------------------------------------------------------------

def _prepare_text(record: UnifiedRecord, strategy: EmbeddingStrategy) -> str:
    """Build the text string to embed for a given strategy.

    Strategy A (answer_only):      student_answer
    Strategy B (question_answer):  question + " " + student_answer
    Strategy C (full_triplet):     question + " " + reference_answer + " " + student_answer
    """
    if strategy == EmbeddingStrategy.ANSWER_ONLY:
        return record.student_answer
    elif strategy == EmbeddingStrategy.QUESTION_ANSWER:
        return record.question + " " + record.student_answer
    elif strategy == EmbeddingStrategy.FULL_TRIPLET:
        return (
            record.question
            + " "
            + record.reference_answer
            + " "
            + record.student_answer
        )
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Grouping by granularity
# ---------------------------------------------------------------------------

def _group_records(
    records: list[UnifiedRecord],
    granularity: Granularity,
) -> dict[str, list[UnifiedRecord]]:
    """Group records by the specified granularity level.

    Returns a dict mapping group_key → list of records.
    """
    groups: dict[str, list[UnifiedRecord]] = {}

    if granularity == Granularity.GLOBAL:
        groups["global"] = list(records)
    elif granularity == Granularity.PER_QUESTION:
        for r in records:
            groups.setdefault(r.question_id, []).append(r)
    elif granularity == Granularity.PER_DOMAIN:
        for r in records:
            groups.setdefault(r.domain, []).append(r)
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")

    return groups


# ---------------------------------------------------------------------------
# Main embedder
# ---------------------------------------------------------------------------

class MisconceptionEmbedder:
    """Embeds filtered misconception records using a configurable SBERT model.

    Args:
        model_name: SBERT model identifier (default: all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _load_sbert(self.model_name)
        return self._model

    def embed(
        self,
        records: Iterable[UnifiedRecord],
        strategy: EmbeddingStrategy,
        granularity: Granularity = Granularity.GLOBAL,
        *,
        filter_records: bool = True,
    ) -> list[EmbeddingResult]:
        """Embed records using the specified strategy and granularity.

        Args:
            records: Input records (will be filtered unless filter_records=False).
            strategy: One of the three embedding strategies (A/B/C).
            granularity: per_question, per_domain, or global.
            filter_records: If True (default), only records with
                label_5way ∈ MISCONCEPTION_LABELS are kept.

        Returns:
            A list of EmbeddingResult, one per group defined by the granularity.
        """
        recs = list(records)
        if filter_records:
            recs = filter_misconception_records(recs)

        if not recs:
            return []

        groups = _group_records(recs, granularity)
        model = self._get_model()
        results: list[EmbeddingResult] = []

        for group_key, group_records in groups.items():
            texts = [_prepare_text(r, strategy) for r in group_records]
            embeddings = model.encode(texts, convert_to_numpy=True)
            results.append(
                EmbeddingResult(
                    embeddings=embeddings,
                    records=group_records,
                    strategy=strategy,
                    granularity=granularity,
                    group_key=group_key,
                )
            )

        return results
