"""Tests for experiments/phase3_misconception.py — Phase 3 misconception mining.

Validates config loading, data loading, gold label encoding, clustering
dispatch, experiment runners, summary building, and result serialization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import UnifiedRecord  # noqa: E402
from src.misconception.embedder import (  # noqa: E402
    EmbeddingResult,
    EmbeddingStrategy,
    Granularity,
)


# ---------------------------------------------------------------------------
# Helpers — create minimal UnifiedRecord instances
# ---------------------------------------------------------------------------

def _make_record(
    sample_id: str = "GEN_0001",
    source_dataset: str = "data_generate",
    split: str = "train",
    question_id: str = "Q1",
    domain: str = "biology",
    question: str = "What is photosynthesis?",
    reference_answer: str = "Plants convert sunlight to energy.",
    student_answer: str = "Plants eat sunlight.",
    label_5way: str | None = "contradictory",
    misconception_tags: list[str] | None = None,
) -> UnifiedRecord:
    return UnifiedRecord(
        sample_id=sample_id,
        source_dataset=source_dataset,
        original_id=sample_id,
        question_id=question_id,
        domain=domain,
        subdomain="general",
        difficulty="medium",
        question=question,
        reference_answer=reference_answer,
        student_answer=student_answer,
        label_5way=label_5way,
        split=split,
        misconception_tags=misconception_tags or [],
    )


def _make_misconception_dataset(n: int = 30) -> list[UnifiedRecord]:
    """Create a dataset with misconception-relevant records."""
    labels = ["contradictory", "partially_correct_incomplete", "irrelevant"]
    tags = [
        ["confuses_photosynthesis_with_respiration"],
        ["thinks_plants_absorb_food_from_soil"],
        ["believes_oxygen_is_main_input"],
    ]
    records = []
    for i in range(n):
        records.append(
            _make_record(
                sample_id=f"GEN_{i:04d}",
                question_id=f"Q{i % 5:03d}",
                domain=["biology", "chemistry", "physics"][i % 3],
                student_answer=f"Student misconception answer {i} about topic",
                label_5way=labels[i % 3],
                misconception_tags=tags[i % 3],
            )
        )
    return records


def _make_mixed_dataset() -> list[UnifiedRecord]:
    """Create a dataset with both correct and incorrect records."""
    records = _make_misconception_dataset(20)
    # Add some correct records that should be filtered out
    for i in range(5):
        records.append(
            _make_record(
                sample_id=f"GEN_CORRECT_{i:04d}",
                student_answer=f"Correct answer {i}",
                label_5way="correct",
            )
        )
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def misconception_config() -> dict:
    """Minimal misconception config for testing."""
    return {
        "seed": 42,
        "sbert_model": "all-MiniLM-L6-v2",
        "embedding_strategies": [
            {"name": "answer_only"},
            {"name": "question_answer"},
            {"name": "full_triplet"},
        ],
        "filter_labels": [
            "partially_correct_incomplete",
            "contradictory",
            "irrelevant",
        ],
        "granularity_levels": ["per_question", "per_domain", "global"],
        "umap": {
            "n_components": 5,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
        },
        "kmeans": {"n_clusters": 3, "n_init": 10, "max_iter": 300},
        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3,
            "cluster_selection_method": "eom",
        },
        "ctfidf": {"top_n_keywords": 5},
    }


@pytest.fixture
def misconception_records() -> list[UnifiedRecord]:
    return _make_misconception_dataset(30)


# ---------------------------------------------------------------------------
# Tests: Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_config_from_yaml(self, tmp_path):
        from experiments.phase3_misconception import load_config

        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text("seed: 99\nsbert_model: test-model\n")
        result = load_config(cfg_file)
        assert result["seed"] == 99
        assert result["sbert_model"] == "test-model"

    def test_load_config_missing_file(self, tmp_path):
        from experiments.phase3_misconception import load_config

        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Tests: Data loading
# ---------------------------------------------------------------------------

class TestLoadUnifiedRecords:
    def test_load_from_jsonl(self, tmp_path):
        from experiments.phase3_misconception import load_unified_records
        import dataclasses

        rec = _make_record()
        rec_dict = dataclasses.asdict(rec)
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(json.dumps(rec_dict) + "\n")

        records = load_unified_records(tmp_path)
        assert len(records) == 1
        assert records[0].sample_id == rec.sample_id

    def test_load_skips_malformed_lines(self, tmp_path):
        from experiments.phase3_misconception import load_unified_records

        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text("not valid json\n")

        records = load_unified_records(tmp_path)
        assert len(records) == 0

    def test_load_empty_dir(self, tmp_path):
        from experiments.phase3_misconception import load_unified_records

        records = load_unified_records(tmp_path)
        assert records == []


# ---------------------------------------------------------------------------
# Tests: Gold label encoding
# ---------------------------------------------------------------------------

class TestEncodeGoldLabels:
    def test_encode_with_tags(self, misconception_records):
        from experiments.phase3_misconception import encode_gold_labels

        labels = encode_gold_labels(misconception_records)
        assert labels is not None
        assert len(labels) == len(misconception_records)
        # Should have 3 unique tags
        unique = set(labels)
        assert len(unique) == 3

    def test_encode_no_tags(self):
        from experiments.phase3_misconception import encode_gold_labels

        records = [
            _make_record(sample_id=f"GEN_{i:04d}", misconception_tags=[])
            for i in range(5)
        ]
        labels = encode_gold_labels(records)
        assert labels is None

    def test_encode_partial_tags(self):
        from experiments.phase3_misconception import encode_gold_labels

        records = [
            _make_record(sample_id="GEN_0001", misconception_tags=["tag_a"]),
            _make_record(sample_id="GEN_0002", misconception_tags=[]),
            _make_record(sample_id="GEN_0003", misconception_tags=["tag_b"]),
        ]
        labels = encode_gold_labels(records)
        assert labels is not None
        assert len(labels) == 3
        assert labels[1] == -1  # No tag → noise
        assert labels[0] != labels[2]  # Different tags


# ---------------------------------------------------------------------------
# Tests: Clustering dispatch
# ---------------------------------------------------------------------------

class TestRunClustering:
    def test_kmeans_clustering(self, misconception_config):
        from experiments.phase3_misconception import run_clustering

        rng = np.random.RandomState(42)
        embeddings = rng.randn(30, 10).astype(np.float32)
        records = _make_misconception_dataset(30)

        emb_result = EmbeddingResult(
            embeddings=embeddings,
            records=records,
            strategy=EmbeddingStrategy.ANSWER_ONLY,
            granularity=Granularity.GLOBAL,
            group_key="global",
        )

        result = run_clustering(emb_result, "kmeans", misconception_config)
        assert result["method"] == "kmeans"
        assert result["n_clusters"] == 3
        assert "keywords" in result
        assert "intrinsic" in result
        assert "extrinsic" in result
        assert "error" not in result

    def test_kmeans_too_few_samples(self, misconception_config):
        from experiments.phase3_misconception import run_clustering

        embeddings = np.array([[1.0, 2.0]])
        records = [_make_record()]

        emb_result = EmbeddingResult(
            embeddings=embeddings,
            records=records,
            strategy=EmbeddingStrategy.ANSWER_ONLY,
            granularity=Granularity.GLOBAL,
            group_key="global",
        )

        result = run_clustering(emb_result, "kmeans", misconception_config)
        assert "error" in result

    def test_unknown_method(self, misconception_config):
        from experiments.phase3_misconception import run_clustering

        embeddings = np.array([[1.0, 2.0]])
        records = [_make_record()]

        emb_result = EmbeddingResult(
            embeddings=embeddings,
            records=records,
            strategy=EmbeddingStrategy.ANSWER_ONLY,
            granularity=Granularity.GLOBAL,
            group_key="global",
        )

        result = run_clustering(emb_result, "unknown_method", misconception_config)
        assert "error" in result

    def test_extrinsic_metrics_present(self, misconception_config):
        """Extrinsic metrics are computed when gold labels are available."""
        from experiments.phase3_misconception import run_clustering

        rng = np.random.RandomState(42)
        embeddings = rng.randn(30, 10).astype(np.float32)
        records = _make_misconception_dataset(30)

        emb_result = EmbeddingResult(
            embeddings=embeddings,
            records=records,
            strategy=EmbeddingStrategy.ANSWER_ONLY,
            granularity=Granularity.GLOBAL,
            group_key="global",
        )

        result = run_clustering(emb_result, "kmeans", misconception_config)
        assert "extrinsic" in result
        extrinsic = result["extrinsic"]
        assert "nmi" in extrinsic
        assert "ari" in extrinsic
        assert "purity" in extrinsic
        assert "v_measure" in extrinsic

    def test_no_gold_labels(self, misconception_config):
        """Extrinsic metrics report error when no gold labels available."""
        from experiments.phase3_misconception import run_clustering

        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 5).astype(np.float32)
        records = [
            _make_record(
                sample_id=f"GEN_{i:04d}",
                misconception_tags=[],
                label_5way="contradictory",
            )
            for i in range(10)
        ]

        emb_result = EmbeddingResult(
            embeddings=embeddings,
            records=records,
            strategy=EmbeddingStrategy.ANSWER_ONLY,
            granularity=Granularity.GLOBAL,
            group_key="global",
        )

        cfg = dict(misconception_config)
        cfg["kmeans"]["n_clusters"] = 3
        result = run_clustering(emb_result, "kmeans", cfg)
        assert result["extrinsic"]["error"] == "No gold misconception_tags available"


# ---------------------------------------------------------------------------
# Tests: Experiment runners
# ---------------------------------------------------------------------------

class TestRunGlobalExperiments:
    def test_global_experiments_with_mock_embedder(self, misconception_config):
        """run_global_experiments returns results for each strategy × method."""
        from experiments.phase3_misconception import run_global_experiments

        records = _make_misconception_dataset(30)
        rng = np.random.RandomState(42)

        # Create a mock embedder that returns fake embeddings
        mock_embedder = MagicMock()

        def mock_embed(recs, strategy, granularity):
            embeddings = rng.randn(len(recs), 10).astype(np.float32)
            return [
                EmbeddingResult(
                    embeddings=embeddings,
                    records=list(recs),
                    strategy=strategy,
                    granularity=granularity,
                    group_key="global",
                )
            ]

        mock_embedder.embed.side_effect = mock_embed

        results = run_global_experiments(records, mock_embedder, misconception_config)

        assert isinstance(results, list)
        assert len(results) > 0

        # Should have results for 3 strategies × 3 methods = 9
        # (some may fail due to sample size for UMAP/HDBSCAN)
        strategies_seen = {r["strategy"] for r in results}
        methods_seen = {r["method"] for r in results}
        assert "answer_only" in strategies_seen
        assert "kmeans" in methods_seen

        for r in results:
            assert r["experiment"] == "global_grid"

    def test_global_experiments_empty_records(self, misconception_config):
        """Returns empty list when no records are available."""
        from experiments.phase3_misconception import run_global_experiments

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = []

        results = run_global_experiments([], mock_embedder, misconception_config)
        assert results == []


class TestRunGranularityExperiments:
    def test_granularity_experiments_with_mock_embedder(
        self, misconception_config
    ):
        """run_granularity_experiments returns results for per-question/domain."""
        from experiments.phase3_misconception import run_granularity_experiments

        records = _make_misconception_dataset(30)
        rng = np.random.RandomState(42)

        mock_embedder = MagicMock()

        def mock_embed(recs, strategy, granularity):
            # Return 2 groups for per-question/per-domain
            half = len(recs) // 2
            if half < 2:
                return []
            return [
                EmbeddingResult(
                    embeddings=rng.randn(half, 10).astype(np.float32),
                    records=list(recs[:half]),
                    strategy=strategy,
                    granularity=granularity,
                    group_key="group_0",
                ),
                EmbeddingResult(
                    embeddings=rng.randn(len(recs) - half, 10).astype(np.float32),
                    records=list(recs[half:]),
                    strategy=strategy,
                    granularity=granularity,
                    group_key="group_1",
                ),
            ]

        mock_embedder.embed.side_effect = mock_embed

        results = run_granularity_experiments(
            records, mock_embedder, misconception_config
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for r in results:
            assert r["experiment"] == "granularity"


# ---------------------------------------------------------------------------
# Tests: Summary building
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_build_summary_with_results(self):
        from experiments.phase3_misconception import build_summary

        results = [
            {
                "experiment": "global_grid",
                "strategy": "answer_only",
                "method": "kmeans",
                "n_clusters": 3,
                "extrinsic": {
                    "nmi": 0.5,
                    "ari": 0.3,
                    "purity": 0.7,
                    "v_measure": 0.45,
                },
                "keywords": {"0": ["word1", "word2"], "1": ["word3"]},
            },
            {
                "experiment": "global_grid",
                "strategy": "question_answer",
                "method": "kmeans",
                "extrinsic": {
                    "nmi": 0.6,
                    "ari": 0.4,
                    "purity": 0.8,
                    "v_measure": 0.55,
                },
                "keywords": {"0": ["word4"]},
                "n_clusters": 3,
            },
            {
                "experiment": "granularity",
                "strategy": "answer_only",
                "method": "kmeans",
                "error": "Too few samples",
            },
        ]

        summary = build_summary(results)
        assert summary["total_experiments"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["best_by_nmi"]["strategy"] == "question_answer"
        assert summary["best_by_nmi"]["nmi"] == 0.6
        assert summary["best_by_ari"]["ari"] == 0.4
        assert len(summary["keyword_summaries"]) == 2

    def test_build_summary_empty(self):
        from experiments.phase3_misconception import build_summary

        summary = build_summary([])
        assert summary["total_experiments"] == 0
        assert summary["successful"] == 0
        assert summary["failed"] == 0
        assert "best_by_nmi" not in summary


# ---------------------------------------------------------------------------
# Tests: Result saving
# ---------------------------------------------------------------------------

class TestResultSaving:
    def test_save_results_creates_json(self, tmp_path):
        from src.evaluation.reporting import save_results

        results = {"experiments": [{"method": "kmeans", "nmi": 0.5}]}
        path = save_results(results, "phase3_test", results_dir=str(tmp_path))

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert data["results"]["experiments"][0]["method"] == "kmeans"

    def test_results_dir_created_if_missing(self, tmp_path):
        from src.evaluation.reporting import save_results

        results_dir = tmp_path / "new_dir" / "phase3"
        save_results({"test": True}, "output", results_dir=str(results_dir))
        assert (results_dir / "output.json").exists()


# ---------------------------------------------------------------------------
# Tests: Strategy and granularity maps
# ---------------------------------------------------------------------------

class TestMaps:
    def test_strategy_map_completeness(self):
        from experiments.phase3_misconception import STRATEGY_MAP

        assert "answer_only" in STRATEGY_MAP
        assert "question_answer" in STRATEGY_MAP
        assert "full_triplet" in STRATEGY_MAP
        assert STRATEGY_MAP["answer_only"] == EmbeddingStrategy.ANSWER_ONLY
        assert STRATEGY_MAP["question_answer"] == EmbeddingStrategy.QUESTION_ANSWER
        assert STRATEGY_MAP["full_triplet"] == EmbeddingStrategy.FULL_TRIPLET

    def test_granularity_map_completeness(self):
        from experiments.phase3_misconception import GRANULARITY_MAP

        assert "per_question" in GRANULARITY_MAP
        assert "per_domain" in GRANULARITY_MAP
        assert "global" in GRANULARITY_MAP

    def test_clustering_methods_list(self):
        from experiments.phase3_misconception import CLUSTERING_METHODS

        assert "kmeans" in CLUSTERING_METHODS
        assert "umap_hdbscan" in CLUSTERING_METHODS
        assert "bertopic" in CLUSTERING_METHODS
