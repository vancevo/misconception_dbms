"""Integration tests for the misconception clustering pipeline.

Tests cover:
  - KMeans clustering on a small embedding set
  - c-TF-IDF keyword extraction
  - UMAP + HDBSCAN (mocked) and BERTopic-style pipeline (mocked)
  - Edge cases: single cluster, all noise

UMAP and HDBSCAN are mocked because they are optional heavy
dependencies. KMeans and c-TF-IDF use real sklearn.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.misconception.clustering import (
    ClusteringMethod,
    ClusterResult,
    cluster_bertopic,
    cluster_hdbscan,
    cluster_kmeans,
    cluster_umap_hdbscan,
    extract_ctfidf_keywords,
    reduce_umap,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_embeddings(
    n_samples: int = 30,
    dim: int = 16,
    n_blobs: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Create a small synthetic embedding set with clear clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_blobs, dim)) * 5
    samples_per = n_samples // n_blobs
    parts = []
    for c in centers:
        pts = c + rng.standard_normal((samples_per, dim)) * 0.3
        parts.append(pts)
    # Handle remainder
    remainder = n_samples - samples_per * n_blobs
    if remainder > 0:
        pts = centers[0] + rng.standard_normal((remainder, dim)) * 0.3
        parts.append(pts)
    return np.vstack(parts)


def _make_documents(n: int = 30) -> list[str]:
    """Create simple documents aligned with 3 clusters."""
    docs = []
    topics = [
        "photosynthesis plants sunlight chlorophyll energy",
        "gravity force mass acceleration newton",
        "mitosis cell division chromosome nucleus",
    ]
    per = n // 3
    for i, topic in enumerate(topics):
        for j in range(per):
            docs.append(f"{topic} sample {i}_{j}")
    remainder = n - per * 3
    for j in range(remainder):
        docs.append(f"{topics[0]} extra {j}")
    return docs


# ------------------------------------------------------------------
# 18.1 — KMeans clustering
# ------------------------------------------------------------------

class TestClusterKMeans:
    def test_produces_correct_number_of_clusters(self):
        emb = _make_embeddings(n_samples=30, n_blobs=3)
        result = cluster_kmeans(emb, n_clusters=3)
        assert result.n_clusters == 3
        assert result.method == ClusteringMethod.KMEANS

    def test_labels_shape_matches_input(self):
        emb = _make_embeddings(n_samples=20, n_blobs=2)
        result = cluster_kmeans(emb, n_clusters=2)
        assert result.labels.shape == (20,)

    def test_all_samples_assigned(self):
        emb = _make_embeddings(n_samples=15, n_blobs=3)
        result = cluster_kmeans(emb, n_clusters=3)
        unique = set(result.labels)
        # All labels should be non-negative integers
        assert all(lbl >= 0 for lbl in unique)

    def test_embeddings_stored_in_result(self):
        emb = _make_embeddings(n_samples=10, n_blobs=2)
        result = cluster_kmeans(emb, n_clusters=2)
        assert np.array_equal(result.embeddings, emb)

    def test_reproducible_with_same_seed(self):
        emb = _make_embeddings(n_samples=30, n_blobs=3)
        r1 = cluster_kmeans(emb, n_clusters=3, random_state=0)
        r2 = cluster_kmeans(emb, n_clusters=3, random_state=0)
        assert np.array_equal(r1.labels, r2.labels)

    def test_single_cluster(self):
        emb = _make_embeddings(n_samples=10, n_blobs=1)
        result = cluster_kmeans(emb, n_clusters=1)
        assert result.n_clusters == 1
        assert all(lbl == 0 for lbl in result.labels)


# ------------------------------------------------------------------
# 18.4 — c-TF-IDF keyword extraction
# ------------------------------------------------------------------

class TestExtractCtfidfKeywords:
    def test_returns_keywords_per_cluster(self):
        docs = _make_documents(30)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        kw = extract_ctfidf_keywords(docs, labels, top_n=5)
        assert set(kw.keys()) == {0, 1, 2}
        for cid in kw:
            assert len(kw[cid]) == 5

    def test_keywords_are_strings(self):
        docs = _make_documents(30)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        kw = extract_ctfidf_keywords(docs, labels, top_n=3)
        for cid in kw:
            for word in kw[cid]:
                assert isinstance(word, str)

    def test_noise_excluded(self):
        docs = ["hello world", "foo bar", "baz qux"]
        labels = np.array([-1, 0, 0])
        kw = extract_ctfidf_keywords(docs, labels, top_n=2)
        assert -1 not in kw
        assert 0 in kw

    def test_all_noise_returns_empty(self):
        docs = ["a b c", "d e f"]
        labels = np.array([-1, -1])
        kw = extract_ctfidf_keywords(docs, labels, top_n=3)
        assert kw == {}

    def test_top_n_respected(self):
        docs = _make_documents(30)
        labels = np.array([0] * 15 + [1] * 15)
        kw = extract_ctfidf_keywords(docs, labels, top_n=2)
        for cid in kw:
            assert len(kw[cid]) == 2

    def test_cluster_specific_terms_ranked_high(self):
        """Cluster-specific words should appear in top keywords."""
        docs = (
            ["photosynthesis plants chlorophyll"] * 10
            + ["gravity force acceleration"] * 10
        )
        labels = np.array([0] * 10 + [1] * 10)
        kw = extract_ctfidf_keywords(docs, labels, top_n=3)
        # Cluster 0 should have plant-related terms
        assert any(
            w in kw[0]
            for w in ("photosynthesis", "plants", "chlorophyll")
        )
        # Cluster 1 should have physics-related terms
        assert any(
            w in kw[1]
            for w in ("gravity", "force", "acceleration")
        )


# ------------------------------------------------------------------
# 18.2 — UMAP + HDBSCAN (mocked external deps)
# ------------------------------------------------------------------

def _mock_umap_module():
    """Return a mock umap module with a UMAP class."""
    mock_mod = MagicMock()

    class FakeUMAP:
        def __init__(self, **kwargs):
            self.n_components = kwargs.get("n_components", 5)

        def fit_transform(self, X):
            rng = np.random.default_rng(0)
            return rng.standard_normal(
                (X.shape[0], self.n_components)
            )

    mock_mod.UMAP = FakeUMAP
    return mock_mod


def _mock_hdbscan_module(n_clusters: int = 2):
    """Return a mock hdbscan module."""
    mock_mod = MagicMock()

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            self._n_clusters = n_clusters

        def fit_predict(self, X):
            n = X.shape[0]
            per = n // self._n_clusters
            labels = []
            for c in range(self._n_clusters):
                if c < self._n_clusters - 1:
                    count = per
                else:
                    count = n - per * (self._n_clusters - 1)
                labels.extend([c] * count)
            return np.array(labels)

    mock_mod.HDBSCAN = FakeHDBSCAN
    return mock_mod


class TestClusterUmapHdbscan:
    @patch(
        "src.misconception.clustering._import_hdbscan",
        return_value=_mock_hdbscan_module(2),
    )
    @patch(
        "src.misconception.clustering._import_umap",
        return_value=_mock_umap_module(),
    )
    def test_produces_cluster_result(self, _umap, _hdb):
        emb = _make_embeddings(n_samples=20, dim=16)
        result = cluster_umap_hdbscan(emb, n_components=5)
        assert isinstance(result, ClusterResult)
        assert result.method == ClusteringMethod.HDBSCAN
        assert result.labels.shape == (20,)

    @patch(
        "src.misconception.clustering._import_hdbscan",
        return_value=_mock_hdbscan_module(3),
    )
    @patch(
        "src.misconception.clustering._import_umap",
        return_value=_mock_umap_module(),
    )
    def test_reduced_embeddings_stored(self, _umap, _hdb):
        emb = _make_embeddings(n_samples=15, dim=32)
        result = cluster_umap_hdbscan(
            emb, n_components=5
        )
        # Reduced embeddings should have n_components columns
        assert result.embeddings.shape == (15, 5)


class TestReduceUmap:
    @patch(
        "src.misconception.clustering._import_umap",
        return_value=_mock_umap_module(),
    )
    def test_output_shape(self, _umap):
        emb = _make_embeddings(n_samples=10, dim=32)
        reduced = reduce_umap(emb, n_components=3)
        assert reduced.shape == (10, 3)


class TestClusterHdbscan:
    @patch(
        "src.misconception.clustering._import_hdbscan",
        return_value=_mock_hdbscan_module(2),
    )
    def test_labels_produced(self, _hdb):
        emb = _make_embeddings(n_samples=10, dim=5)
        result = cluster_hdbscan(emb)
        assert result.labels.shape == (10,)
        assert result.method == ClusteringMethod.HDBSCAN


# ------------------------------------------------------------------
# 18.3 — BERTopic-style pipeline (mocked)
# ------------------------------------------------------------------

class TestClusterBertopic:
    @patch(
        "src.misconception.clustering._import_hdbscan",
        return_value=_mock_hdbscan_module(2),
    )
    @patch(
        "src.misconception.clustering._import_umap",
        return_value=_mock_umap_module(),
    )
    def test_full_pipeline_produces_keywords(self, _umap, _hdb):
        emb = _make_embeddings(n_samples=20, dim=16)
        docs = _make_documents(20)
        result = cluster_bertopic(
            emb,
            docs,
            n_components=5,
            top_n_keywords=3,
        )
        assert result.method == ClusteringMethod.BERTOPIC
        assert result.labels.shape == (20,)
        # Keywords should be populated for non-noise clusters
        assert len(result.keywords) > 0
        for cid, words in result.keywords.items():
            assert len(words) <= 3
            assert all(isinstance(w, str) for w in words)

    @patch(
        "src.misconception.clustering._import_hdbscan",
        return_value=_mock_hdbscan_module(3),
    )
    @patch(
        "src.misconception.clustering._import_umap",
        return_value=_mock_umap_module(),
    )
    def test_reduced_embeddings_in_result(self, _umap, _hdb):
        emb = _make_embeddings(n_samples=30, dim=16)
        docs = _make_documents(30)
        result = cluster_bertopic(
            emb, docs, n_components=5
        )
        assert result.embeddings.shape == (30, 5)


# ------------------------------------------------------------------
# 18.5 — Integration: KMeans + c-TF-IDF end-to-end
# ------------------------------------------------------------------

class TestKMeansWithKeywords:
    """End-to-end: cluster with KMeans, then extract keywords."""

    def test_kmeans_then_ctfidf(self):
        emb = _make_embeddings(n_samples=30, n_blobs=3)
        docs = _make_documents(30)
        result = cluster_kmeans(emb, n_clusters=3)
        kw = extract_ctfidf_keywords(
            docs, result.labels, top_n=5
        )
        result.keywords = kw
        # Should have keywords for each cluster
        assert len(kw) == 3
        for cid in kw:
            assert len(kw[cid]) == 5


# ------------------------------------------------------------------
# Lazy import error tests
# ------------------------------------------------------------------

class TestLazyImportErrors:
    def test_umap_import_error(self):
        with patch.dict("sys.modules", {"umap": None}):
            with pytest.raises(ImportError, match="umap-learn"):
                _import_umap_real()

    def test_hdbscan_import_error(self):
        with patch.dict("sys.modules", {"hdbscan": None}):
            with pytest.raises(ImportError, match="hdbscan"):
                _import_hdbscan_real()


def _import_umap_real():
    """Call the real lazy import to test error handling."""
    from src.misconception.clustering import _import_umap
    return _import_umap()


def _import_hdbscan_real():
    """Call the real lazy import to test error handling."""
    from src.misconception.clustering import _import_hdbscan
    return _import_hdbscan()
