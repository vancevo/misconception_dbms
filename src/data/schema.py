"""Unified record schema for the ASAG Research Framework.

All four source datasets (SciEntsBank, MohlerASAG, Data_Generate,
Data_Scraping) are converted into this canonical format before any
downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_SOURCE_DATASETS = frozenset(
    {"scientsbank", "mohler", "data_generate", "data_scraping"}
)

VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "unknown"})


@dataclass
class UnifiedRecord:
    """Canonical data structure for a single student-answer sample."""

    # ── Identity ──────────────────────────────────────────────────────
    sample_id: str
    source_dataset: str
    original_id: str
    question_id: str

    # ── Domain ────────────────────────────────────────────────────────
    domain: str
    subdomain: str
    difficulty: str  # "easy" | "medium" | "hard" | "unknown"

    # ── Core triplet ──────────────────────────────────────────────────
    question: str
    reference_answer: str
    student_answer: str
    alternative_reference_answers: list[str] = field(default_factory=list)

    # ── Grading labels ────────────────────────────────────────────────
    score_raw: float | None = None
    score_normalized: float | None = None
    label_2way: str | None = None
    label_3way: str | None = None
    label_5way: str | None = None

    # ── Concept-level annotations ─────────────────────────────────────
    key_concepts: list[str] = field(default_factory=list)
    misconception_tags: list[str] = field(default_factory=list)
    misconception_inventory: list[dict] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    extra_incorrect_claims: list[str] = field(default_factory=list)

    # ── Feedback ──────────────────────────────────────────────────────
    feedback_short: str | None = None
    feedback_detailed: str | None = None
    feedback_type: str | None = None
    feedback_tone: str | None = None

    # ── Splits and metadata ───────────────────────────────────────────
    split: str = ""
    is_human_annotated: bool = False
    is_synthetic: bool = False
    is_adversarial: bool = False
    perturbation_type: str | None = None
    adversarial_variant_of: str | None = None
    student_answer_style: str | None = None
    annotation_confidence: float | None = None

    # ── Usability flags ───────────────────────────────────────────────
    usable_for_grading: bool = True
    usable_for_feedback: bool = True
    usable_for_misconception_mining: bool = True
    usable_for_robustness_eval: bool = True

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if self.source_dataset not in VALID_SOURCE_DATASETS:
            raise ValueError(
                f"source_dataset must be one of {sorted(VALID_SOURCE_DATASETS)}, "
                f"got {self.source_dataset!r}"
            )

        if self.score_normalized is not None:
            if not (0.0 <= self.score_normalized <= 1.0):
                raise ValueError(
                    f"score_normalized must be in [0.0, 1.0], "
                    f"got {self.score_normalized}"
                )

        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of {sorted(VALID_DIFFICULTIES)}, "
                f"got {self.difficulty!r}"
            )
