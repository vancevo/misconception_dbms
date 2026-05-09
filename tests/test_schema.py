"""Tests for the UnifiedRecord schema.

Covers construction, field types, validation, and property-based tests.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.schema import VALID_SOURCE_DATASETS, UnifiedRecord


# ── Helpers ───────────────────────────────────────────────────────────

def _make_record(**overrides) -> UnifiedRecord:
    """Create a minimal valid UnifiedRecord with optional overrides."""
    defaults = dict(
        sample_id="SEB_0001",
        source_dataset="scientsbank",
        original_id="orig_1",
        question_id="q_1",
        domain="science",
        subdomain="biology",
        difficulty="easy",
        question="What is photosynthesis?",
        reference_answer="The process by which plants convert light to energy.",
        student_answer="Plants use sunlight to make food.",
    )
    defaults.update(overrides)
    return UnifiedRecord(**defaults)


# ── 2.3  Unit tests: construction, field types, validation ────────────

class TestUnifiedRecordConstruction:
    """Verify basic construction and field types."""

    def test_minimal_construction(self):
        rec = _make_record()
        assert rec.sample_id == "SEB_0001"
        assert rec.source_dataset == "scientsbank"
        assert rec.difficulty == "easy"

    def test_default_list_fields_are_empty(self):
        rec = _make_record()
        assert rec.alternative_reference_answers == []
        assert rec.key_concepts == []
        assert rec.misconception_tags == []
        assert rec.misconception_inventory == []
        assert rec.missing_concepts == []
        assert rec.extra_incorrect_claims == []

    def test_default_optional_fields_are_none(self):
        rec = _make_record()
        assert rec.score_raw is None
        assert rec.score_normalized is None
        assert rec.label_2way is None
        assert rec.label_3way is None
        assert rec.label_5way is None
        assert rec.feedback_short is None
        assert rec.feedback_detailed is None

    def test_default_boolean_flags(self):
        rec = _make_record()
        assert rec.is_human_annotated is False
        assert rec.is_synthetic is False
        assert rec.is_adversarial is False
        assert rec.usable_for_grading is True
        assert rec.usable_for_feedback is True
        assert rec.usable_for_misconception_mining is True
        assert rec.usable_for_robustness_eval is True

    def test_all_four_source_datasets_accepted(self):
        for src in VALID_SOURCE_DATASETS:
            rec = _make_record(source_dataset=src)
            assert rec.source_dataset == src

    def test_all_difficulty_levels_accepted(self):
        for diff in ("easy", "medium", "hard", "unknown"):
            rec = _make_record(difficulty=diff)
            assert rec.difficulty == diff

    def test_score_normalized_boundary_values(self):
        rec0 = _make_record(score_normalized=0.0)
        assert rec0.score_normalized == 0.0
        rec1 = _make_record(score_normalized=1.0)
        assert rec1.score_normalized == 1.0
        rec_mid = _make_record(score_normalized=0.5)
        assert rec_mid.score_normalized == 0.5

    def test_list_fields_independent_across_instances(self):
        r1 = _make_record()
        r2 = _make_record()
        r1.key_concepts.append("photosynthesis")
        assert r2.key_concepts == []


class TestUnifiedRecordValidation:
    """Verify __post_init__ validation rules."""

    def test_invalid_source_dataset_raises(self):
        with pytest.raises(ValueError, match="source_dataset"):
            _make_record(source_dataset="invalid_source")

    def test_score_normalized_below_zero_raises(self):
        with pytest.raises(ValueError, match="score_normalized"):
            _make_record(score_normalized=-0.1)

    def test_score_normalized_above_one_raises(self):
        with pytest.raises(ValueError, match="score_normalized"):
            _make_record(score_normalized=1.01)

    def test_score_normalized_none_is_valid(self):
        rec = _make_record(score_normalized=None)
        assert rec.score_normalized is None

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError, match="difficulty"):
            _make_record(difficulty="extreme")


# ── 2.4  Property 5: Unique sample IDs across all sources ────────────
# Feature: asag-research-framework, Property 5: For any batch of records
# from all sources, all sample_id values are unique

_source_prefixes = {
    "scientsbank": "SEB",
    "mohler": "MOH",
    "data_generate": "GEN",
    "data_scraping": "SCR",
}

_record_strategy = st.fixed_dictionaries(
    {
        "source_dataset": st.sampled_from(sorted(VALID_SOURCE_DATASETS)),
        "index": st.integers(min_value=0, max_value=99_999),
    }
)


def _build_record_from_strategy(source_dataset: str, index: int) -> UnifiedRecord:
    prefix = _source_prefixes[source_dataset]
    sample_id = f"{prefix}_{index:05d}"
    return _make_record(
        sample_id=sample_id,
        source_dataset=source_dataset,
        original_id=f"orig_{index}",
        question_id=f"q_{index}",
    )


@given(
    batch=st.lists(
        st.tuples(
            st.sampled_from(sorted(VALID_SOURCE_DATASETS)),
            st.integers(min_value=0, max_value=99_999),
        ),
        min_size=1,
        max_size=50,
        unique=True,  # ensure (source, index) pairs are unique
    )
)
@settings(max_examples=100)
def test_property5_unique_sample_ids(batch):
    """**Validates: Requirements 1.5**

    For any batch of records from all sources, all sample_id values are unique.
    """
    # Feature: asag-research-framework, Property 5: For any batch of records
    # from all sources, all sample_id values are unique
    records = [_build_record_from_strategy(src, idx) for src, idx in batch]
    ids = [r.sample_id for r in records]
    assert len(ids) == len(set(ids)), f"Duplicate sample_ids found: {ids}"


# ── 2.5  Property 6: Data_Scraping usability flags all false ─────────
# Feature: asag-research-framework, Property 6: For any Data_Scraping record,
# all four usability flags are false

@given(
    index=st.integers(min_value=0, max_value=99_999),
    question=st.text(min_size=1, max_size=200),
    reference=st.text(min_size=1, max_size=200),
)
@settings(max_examples=100)
def test_property6_data_scraping_usability_flags(index, question, reference):
    """**Validates: Requirements 1.4**

    For any Data_Scraping record, all four usability flags are false.
    """
    # Feature: asag-research-framework, Property 6: For any Data_Scraping record,
    # all four usability flags are false
    rec = UnifiedRecord(
        sample_id=f"SCR_{index:05d}",
        source_dataset="data_scraping",
        original_id=f"scr_orig_{index}",
        question_id=f"scr_q_{index}",
        domain="science",
        subdomain="general",
        difficulty="unknown",
        question=question,
        reference_answer=reference,
        student_answer="",
        usable_for_grading=False,
        usable_for_feedback=False,
        usable_for_misconception_mining=False,
        usable_for_robustness_eval=False,
    )
    assert rec.usable_for_grading is False
    assert rec.usable_for_feedback is False
    assert rec.usable_for_misconception_mining is False
    assert rec.usable_for_robustness_eval is False
