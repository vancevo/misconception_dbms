"""Unit tests for the DataLoader public API (src/data/dataset.py).

All tests use in-memory fixture records — no JSONL files needed.
"""

from __future__ import annotations

import pytest

from src.data.dataset import DataLoader
from src.data.schema import UnifiedRecord


# ── Fixture helpers ───────────────────────────────────────────────────


def _rec(
    sample_id: str = "GEN_00001",
    source_dataset: str = "data_generate",
    split: str = "train",
    question: str = "What is X?",
    reference_answer: str = "X is Y.",
    student_answer: str = "X is Y.",
    domain: str = "science",
    label_5way: str | None = "correct",
    label_3way: str | None = "correct",
    label_2way: str | None = "correct",
    is_adversarial: bool = False,
    usable_for_grading: bool = True,
    score_normalized: float | None = None,
    **kwargs,
) -> UnifiedRecord:
    """Shorthand factory for test records."""
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset=source_dataset,
        original_id=sample_id,
        question_id=f"q_{sample_id}",
        domain=domain,
        subdomain="general",
        difficulty="unknown",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        split=split,
        label_5way=label_5way,
        label_3way=label_3way,
        label_2way=label_2way,
        is_adversarial=is_adversarial,
        usable_for_grading=usable_for_grading,
        score_normalized=score_normalized,
        **kwargs,
    )


def _make_fixture_records() -> list[UnifiedRecord]:
    """Build a small set of records spanning two sources and multiple splits."""
    return [
        # SciEntsBank records
        _rec(sample_id="SEB_00001", source_dataset="scientsbank", split="train",
             domain="science", label_3way="correct", label_2way="correct"),
        _rec(sample_id="SEB_00002", source_dataset="scientsbank", split="train",
             domain="biology", label_3way="incorrect", label_2way="incorrect",
             is_adversarial=True),
        _rec(sample_id="SEB_00003", source_dataset="scientsbank", split="test_ua",
             domain="science", label_3way="correct", label_2way="correct"),
        _rec(sample_id="SEB_00004", source_dataset="scientsbank", split="test_uq",
             domain="science", label_3way="partially_correct",
             label_2way="incorrect"),
        # Data_Generate records
        _rec(sample_id="GEN_00001", source_dataset="data_generate", split="train",
             domain="math", label_3way="correct", label_2way="correct",
             usable_for_grading=True, score_normalized=0.9),
        _rec(sample_id="GEN_00002", source_dataset="data_generate", split="train",
             domain="math", label_3way="incorrect", label_2way="incorrect",
             usable_for_grading=False, score_normalized=0.2),
        _rec(sample_id="GEN_00003", source_dataset="data_generate", split="test_adversarial",
             domain="math", label_3way="incorrect", label_2way="incorrect",
             is_adversarial=True),
    ]


@pytest.fixture
def loader() -> DataLoader:
    return DataLoader(_make_fixture_records())


# ── 7.1  get_split basic ─────────────────────────────────────────────


class TestGetSplit:
    def test_returns_correct_records(self, loader: DataLoader):
        recs = loader.get_split("scientsbank", "train")
        assert len(recs) == 2
        assert all(r.source_dataset == "scientsbank" for r in recs)
        assert all(r.split == "train" for r in recs)

    def test_different_split(self, loader: DataLoader):
        recs = loader.get_split("scientsbank", "test_ua")
        assert len(recs) == 1
        assert recs[0].sample_id == "SEB_00003"

    def test_data_generate_split(self, loader: DataLoader):
        recs = loader.get_split("data_generate", "train")
        assert len(recs) == 2


# ── 7.2  Filtering ───────────────────────────────────────────────────


class TestFiltering:
    def test_filter_by_domain(self, loader: DataLoader):
        recs = loader.get_split("scientsbank", "train", filters={"domain": "science"})
        assert len(recs) == 1
        assert recs[0].sample_id == "SEB_00001"

    def test_filter_by_label_3way(self, loader: DataLoader):
        recs = loader.get_split("scientsbank", "train", filters={"label_3way": "incorrect"})
        assert len(recs) == 1
        assert recs[0].sample_id == "SEB_00002"

    def test_filter_by_label_2way(self, loader: DataLoader):
        recs = loader.get_split("data_generate", "train", filters={"label_2way": "correct"})
        assert len(recs) == 1
        assert recs[0].sample_id == "GEN_00001"

    def test_filter_by_is_adversarial(self, loader: DataLoader):
        recs = loader.get_split("scientsbank", "train", filters={"is_adversarial": True})
        assert len(recs) == 1
        assert recs[0].sample_id == "SEB_00002"

    def test_filter_by_usable_for_grading(self, loader: DataLoader):
        recs = loader.get_split("data_generate", "train", filters={"usable_for_grading": False})
        assert len(recs) == 1
        assert recs[0].sample_id == "GEN_00002"

    def test_filter_by_source_dataset(self, loader: DataLoader):
        # source_dataset filter is redundant with get_split's source arg,
        # but should still work
        recs = loader.get_split(
            "scientsbank", "train",
            filters={"source_dataset": "scientsbank"},
        )
        assert len(recs) == 2

    def test_multiple_filters(self, loader: DataLoader):
        recs = loader.get_split(
            "scientsbank", "train",
            filters={"domain": "science", "label_2way": "correct"},
        )
        assert len(recs) == 1
        assert recs[0].sample_id == "SEB_00001"

    def test_no_match_returns_empty(self, loader: DataLoader):
        recs = loader.get_split(
            "scientsbank", "train",
            filters={"domain": "nonexistent"},
        )
        assert recs == []

    def test_none_filters_returns_all(self, loader: DataLoader):
        recs = loader.get_split("scientsbank", "train", filters=None)
        assert len(recs) == 2


# ── 7.3  Cross-dataset merge ─────────────────────────────────────────


class TestCrossDatasetMerge:
    def test_merge_two_sources(self, loader: DataLoader):
        recs = loader.get_merged([
            ("scientsbank", "train"),
            ("data_generate", "train"),
        ])
        assert len(recs) == 4
        sources = {r.source_dataset for r in recs}
        assert sources == {"scientsbank", "data_generate"}

    def test_merge_preserves_source_dataset(self, loader: DataLoader):
        recs = loader.get_merged([
            ("scientsbank", "train"),
            ("data_generate", "train"),
        ])
        for r in recs:
            assert r.source_dataset in ("scientsbank", "data_generate")

    def test_merge_with_filters(self, loader: DataLoader):
        recs = loader.get_merged(
            [("scientsbank", "train"), ("data_generate", "train")],
            filters={"label_2way": "correct"},
        )
        assert len(recs) == 2
        assert all(r.label_2way == "correct" for r in recs)

    def test_merge_single_source(self, loader: DataLoader):
        recs = loader.get_merged([("scientsbank", "test_ua")])
        assert len(recs) == 1

    def test_merge_raises_on_bad_split(self, loader: DataLoader):
        with pytest.raises(ValueError, match="does not exist"):
            loader.get_merged([("scientsbank", "nonexistent")])


# ── 7.4  get_training_batch ──────────────────────────────────────────


class TestGetTrainingBatch:
    def test_yields_triplet_and_label(self, loader: DataLoader):
        batches = list(loader.get_training_batch(
            [("data_generate", "train")],
            label_field="label_3way",
        ))
        assert len(batches) == 2
        for (q, ref, sa), label in batches:
            assert isinstance(q, str)
            assert isinstance(ref, str)
            assert isinstance(sa, str)
            assert label in ("correct", "incorrect")

    def test_score_normalized_label(self, loader: DataLoader):
        batches = list(loader.get_training_batch(
            [("data_generate", "train")],
            label_field="score_normalized",
        ))
        labels = [label for _, label in batches]
        assert 0.9 in labels
        assert 0.2 in labels

    def test_cross_source_batch(self, loader: DataLoader):
        batches = list(loader.get_training_batch(
            [("scientsbank", "train"), ("data_generate", "train")],
            label_field="label_2way",
        ))
        assert len(batches) == 4

    def test_batch_with_filters(self, loader: DataLoader):
        batches = list(loader.get_training_batch(
            [("data_generate", "train")],
            label_field="label_2way",
            filters={"usable_for_grading": True},
        ))
        assert len(batches) == 1
        assert batches[0][1] == "correct"


# ── 7.5  ValueError on missing split ─────────────────────────────────


class TestValueErrorOnMissingSplit:
    def test_get_split_raises(self, loader: DataLoader):
        with pytest.raises(ValueError, match="does not exist") as exc_info:
            loader.get_split("scientsbank", "nonexistent_split")
        # Error message should list available splits
        msg = str(exc_info.value)
        assert "test_ua" in msg
        assert "train" in msg

    def test_get_split_raises_for_unknown_source(self, loader: DataLoader):
        with pytest.raises(ValueError, match="does not exist"):
            loader.get_split("unknown_source", "train")

    def test_get_training_batch_raises(self, loader: DataLoader):
        with pytest.raises(ValueError, match="does not exist"):
            list(loader.get_training_batch(
                [("scientsbank", "bad_split")],
                label_field="label_2way",
            ))

    def test_error_lists_available_splits(self, loader: DataLoader):
        with pytest.raises(ValueError) as exc_info:
            loader.get_split("data_generate", "nonexistent")
        msg = str(exc_info.value)
        assert "train" in msg
        assert "test_adversarial" in msg
