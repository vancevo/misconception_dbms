"""Unit tests for misconception embedding strategies.

All tests mock the SentenceTransformer so no model weights
are downloaded. The mock returns deterministic embeddings.

Covers:
- Record filtering (label_5way inclusion/exclusion)
- Strategy A/B/C text preparation and embedding shape
- Granularity levels: per-question, per-domain, global
- Edge cases: empty input, records with excluded labels
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.data.schema import UnifiedRecord
from src.misconception.embedder import (
    EmbeddingStrategy,
    Granularity,
    MisconceptionEmbedder,
    filter_misconception_records,
    _prepare_text,
    _group_records,
)


# ---------------------------------------------------------------------------
# Mock SBERT model
# ---------------------------------------------------------------------------

EMBED_DIM = 8


def _make_mock_sbert():
    """Return a mock SentenceTransformer with deterministic embeddings."""

    def _encode(texts, convert_to_numpy=True, **kwargs):
        result = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(EMBED_DIM)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec)
        return np.array(result)

    mock = MagicMock()
    mock.encode.side_effect = _encode
    return mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    label_5way: str | None = "contradictory",
    question_id: str = "Q1",
    domain: str = "science",
    question: str = "What is photosynthesis?",
    reference_answer: str = "The process of converting light to energy.",
    student_answer: str = "Plants eat sunlight.",
    sample_id: str = "GEN_0001",
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset="data_generate",
        original_id=sample_id,
        question_id=question_id,
        domain=domain,
        subdomain="biology",
        difficulty="medium",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        label_5way=label_5way,
    )


# ---------------------------------------------------------------------------
# filter_misconception_records tests
# ---------------------------------------------------------------------------

class TestFilterMisconceptionRecords:
    def test_keeps_partially_correct_incomplete(self):
        r = _make_record(label_5way="partially_correct_incomplete")
        assert filter_misconception_records([r]) == [r]

    def test_keeps_contradictory(self):
        r = _make_record(label_5way="contradictory")
        assert filter_misconception_records([r]) == [r]

    def test_keeps_irrelevant(self):
        r = _make_record(label_5way="irrelevant")
        assert filter_misconception_records([r]) == [r]

    def test_excludes_correct(self):
        r = _make_record(label_5way="correct")
        assert filter_misconception_records([r]) == []

    def test_excludes_non_domain(self):
        r = _make_record(label_5way="non_domain")
        assert filter_misconception_records([r]) == []

    def test_excludes_none_label(self):
        r = _make_record(label_5way=None)
        assert filter_misconception_records([r]) == []

    def test_mixed_labels_filters_correctly(self):
        records = [
            _make_record(
                label_5way="correct", sample_id="GEN_0001"
            ),
            _make_record(
                label_5way="contradictory", sample_id="GEN_0002"
            ),
            _make_record(
                label_5way="irrelevant", sample_id="GEN_0003"
            ),
            _make_record(
                label_5way="non_domain", sample_id="GEN_0004"
            ),
            _make_record(
                label_5way="partially_correct_incomplete",
                sample_id="GEN_0005",
            ),
        ]
        filtered = filter_misconception_records(records)
        assert len(filtered) == 3
        ids = {r.sample_id for r in filtered}
        assert ids == {"GEN_0002", "GEN_0003", "GEN_0005"}

    def test_empty_input(self):
        assert filter_misconception_records([]) == []


# ---------------------------------------------------------------------------
# _prepare_text tests
# ---------------------------------------------------------------------------

class TestPrepareText:
    def test_strategy_a_answer_only(self):
        r = _make_record(
            question="Q?",
            reference_answer="Ref",
            student_answer="Stu",
        )
        assert _prepare_text(r, EmbeddingStrategy.ANSWER_ONLY) == "Stu"

    def test_strategy_b_question_answer(self):
        r = _make_record(
            question="Q?",
            reference_answer="Ref",
            student_answer="Stu",
        )
        assert _prepare_text(r, EmbeddingStrategy.QUESTION_ANSWER) == "Q? Stu"

    def test_strategy_c_full_triplet(self):
        r = _make_record(
            question="Q?",
            reference_answer="Ref",
            student_answer="Stu",
        )
        assert _prepare_text(r, EmbeddingStrategy.FULL_TRIPLET) == "Q? Ref Stu"


# ---------------------------------------------------------------------------
# _group_records tests
# ---------------------------------------------------------------------------

class TestGroupRecords:
    def test_global_single_group(self):
        records = [
            _make_record(question_id="Q1", domain="sci", sample_id="GEN_0001"),
            _make_record(question_id="Q2", domain="math", sample_id="GEN_0002"),
        ]
        groups = _group_records(records, Granularity.GLOBAL)
        assert list(groups.keys()) == ["global"]
        assert len(groups["global"]) == 2

    def test_per_question_groups(self):
        records = [
            _make_record(question_id="Q1", sample_id="GEN_0001"),
            _make_record(question_id="Q1", sample_id="GEN_0002"),
            _make_record(question_id="Q2", sample_id="GEN_0003"),
        ]
        groups = _group_records(records, Granularity.PER_QUESTION)
        assert set(groups.keys()) == {"Q1", "Q2"}
        assert len(groups["Q1"]) == 2
        assert len(groups["Q2"]) == 1

    def test_per_domain_groups(self):
        records = [
            _make_record(domain="science", sample_id="GEN_0001"),
            _make_record(domain="science", sample_id="GEN_0002"),
            _make_record(domain="math", sample_id="GEN_0003"),
        ]
        groups = _group_records(records, Granularity.PER_DOMAIN)
        assert set(groups.keys()) == {"science", "math"}
        assert len(groups["science"]) == 2
        assert len(groups["math"]) == 1


# ---------------------------------------------------------------------------
# MisconceptionEmbedder tests
# ---------------------------------------------------------------------------

class TestMisconceptionEmbedder:
    @patch("src.misconception.embedder._load_sbert")
    def test_embed_returns_correct_shape_strategy_a(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="irrelevant",
                sample_id="GEN_0002",
            ),
            _make_record(
                label_5way="partially_correct_incomplete",
                sample_id="GEN_0003",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records, EmbeddingStrategy.ANSWER_ONLY
        )
        assert len(results) == 1  # global granularity → 1 group
        assert results[0].embeddings.shape == (3, EMBED_DIM)
        assert len(results[0].records) == 3

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_returns_correct_shape_strategy_b(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="irrelevant",
                sample_id="GEN_0002",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records, EmbeddingStrategy.QUESTION_ANSWER
        )
        assert len(results) == 1
        assert results[0].embeddings.shape == (2, EMBED_DIM)

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_returns_correct_shape_strategy_c(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0001",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records, EmbeddingStrategy.FULL_TRIPLET
        )
        assert len(results) == 1
        assert results[0].embeddings.shape == (1, EMBED_DIM)

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_filters_out_correct_labels(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="correct",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0002",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records, EmbeddingStrategy.ANSWER_ONLY
        )
        assert len(results) == 1
        assert results[0].embeddings.shape == (1, EMBED_DIM)
        assert results[0].records[0].sample_id == "GEN_0002"

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_empty_after_filtering(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="correct",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="non_domain",
                sample_id="GEN_0002",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records, EmbeddingStrategy.ANSWER_ONLY
        )
        assert results == []

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_per_question_granularity(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                question_id="Q1",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="irrelevant",
                question_id="Q1",
                sample_id="GEN_0002",
            ),
            _make_record(
                label_5way="contradictory",
                question_id="Q2",
                sample_id="GEN_0003",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records,
            EmbeddingStrategy.ANSWER_ONLY,
            Granularity.PER_QUESTION,
        )
        assert len(results) == 2
        keys = {r.group_key for r in results}
        assert keys == {"Q1", "Q2"}
        for res in results:
            if res.group_key == "Q1":
                assert res.embeddings.shape == (2, EMBED_DIM)
            else:
                assert res.embeddings.shape == (1, EMBED_DIM)

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_per_domain_granularity(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                domain="science",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="irrelevant",
                domain="math",
                sample_id="GEN_0002",
            ),
            _make_record(
                label_5way="contradictory",
                domain="math",
                sample_id="GEN_0003",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records,
            EmbeddingStrategy.ANSWER_ONLY,
            Granularity.PER_DOMAIN,
        )
        assert len(results) == 2
        keys = {r.group_key for r in results}
        assert keys == {"science", "math"}

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_result_metadata(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0001",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records,
            EmbeddingStrategy.FULL_TRIPLET,
            Granularity.GLOBAL,
        )
        assert len(results) == 1
        res = results[0]
        assert res.strategy == EmbeddingStrategy.FULL_TRIPLET
        assert res.granularity == Granularity.GLOBAL
        assert res.group_key == "global"

    @patch("src.misconception.embedder._load_sbert")
    def test_embed_no_filter_mode(self, mock_load):
        """All records embedded when filter_records=False."""
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="correct",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0002",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records,
            EmbeddingStrategy.ANSWER_ONLY,
            filter_records=False,
        )
        assert len(results) == 1
        assert results[0].embeddings.shape == (2, EMBED_DIM)

    @patch("src.misconception.embedder._load_sbert")
    def test_strategies_produce_different_texts(
        self, mock_load
    ):
        """Different strategies encode different text."""
        mock_load.return_value = _make_mock_sbert()
        r = _make_record(
            label_5way="contradictory",
            question="What is X?",
            reference_answer="X is Y.",
            student_answer="X is Z.",
            sample_id="GEN_0001",
        )
        embedder = MisconceptionEmbedder()
        res_a = embedder.embed(
            [r], EmbeddingStrategy.ANSWER_ONLY
        )
        res_b = embedder.embed(
            [r], EmbeddingStrategy.QUESTION_ANSWER
        )
        res_c = embedder.embed(
            [r], EmbeddingStrategy.FULL_TRIPLET
        )
        assert res_a[0].embeddings.shape == (1, EMBED_DIM)
        assert res_b[0].embeddings.shape == (1, EMBED_DIM)
        assert res_c[0].embeddings.shape == (1, EMBED_DIM)
        # Embeddings differ because input text differs
        assert not np.allclose(
            res_a[0].embeddings, res_b[0].embeddings
        )
        assert not np.allclose(
            res_b[0].embeddings, res_c[0].embeddings
        )

    @patch("src.misconception.embedder._load_sbert")
    def test_embeddings_are_2d_numpy_array(
        self, mock_load
    ):
        mock_load.return_value = _make_mock_sbert()
        records = [
            _make_record(
                label_5way="contradictory",
                sample_id="GEN_0001",
            ),
            _make_record(
                label_5way="irrelevant",
                sample_id="GEN_0002",
            ),
        ]
        embedder = MisconceptionEmbedder()
        results = embedder.embed(
            records, EmbeddingStrategy.ANSWER_ONLY
        )
        emb = results[0].embeddings
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
