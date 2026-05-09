"""Unit tests for the cluster evaluator module.

Tests intrinsic and extrinsic metrics with known cluster assignments
to verify metric values are in expected ranges and match known results.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.misconception.clustering import ClusterResult, ClusteringMethod
from src.misconception.evaluator import (
    ClusterEvaluation,
    ExtrinsicMetrics,
    IntrinsicMetrics,
    _compute_purity,
    compute_extrinsic_metrics,
    compute_intrinsic_metrics,
    evaluate_clustering,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_cluster_result(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> ClusterResult:
    """Create a minimal ClusterResult for testing."""
    n_clusters = len(set(labels) - {-1})
    return ClusterResult(
        labels=labels,
        method=ClusteringMethod.KMEANS,
        n_clusters=n_clusters,
        embeddings=embeddings,
    )


def _well_separated_data() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated 2-D clusters."""
    rng = np.random.RandomState(42)
    c0 = rng.randn(30, 2) + np.array([5.0, 5.0])
    c1 = rng.randn(30, 2) + np.array([-5.0, -5.0])
    embeddings = np.vstack([c0, c1])
    labels = np.array([0] * 30 + [1] * 30)
    return embeddings, labels


# ------------------------------------------------------------------
# Intrinsic metrics tests
# ------------------------------------------------------------------


class TestIntrinsicMetrics:
    """Tests for compute_intrinsic_metrics."""

    def test_well_separated_clusters_high_silhouette(self):
        embeddings, labels = _well_separated_data()
        metrics = compute_intrinsic_metrics(embeddings, labels)

        assert isinstance(metrics, IntrinsicMetrics)
        # Well-separated clusters should have silhouette close to 1
        assert metrics.silhouette > 0.8
        # Calinski-Harabasz should be positive and large
        assert metrics.calinski_harabasz > 0
        # Davies-Bouldin should be small for well-separated clusters
        assert metrics.davies_bouldin < 0.5

    def test_overlapping_clusters_lower_silhouette(self):
        rng = np.random.RandomState(42)
        # Overlapping clusters centered at (0,0) and (1,0)
        c0 = rng.randn(30, 2) + np.array([0.0, 0.0])
        c1 = rng.randn(30, 2) + np.array([1.0, 0.0])
        embeddings = np.vstack([c0, c1])
        labels = np.array([0] * 30 + [1] * 30)

        metrics = compute_intrinsic_metrics(embeddings, labels)
        # Overlapping clusters should have lower silhouette
        assert metrics.silhouette < 0.8

    def test_noise_labels_excluded(self):
        embeddings, labels = _well_separated_data()
        # Mark some points as noise
        labels_with_noise = labels.copy()
        labels_with_noise[:5] = -1

        metrics = compute_intrinsic_metrics(embeddings, labels_with_noise)
        assert isinstance(metrics, IntrinsicMetrics)
        # Should still compute valid metrics on non-noise points
        assert metrics.silhouette > 0.5

    def test_single_cluster_raises(self):
        embeddings = np.random.randn(20, 2)
        labels = np.zeros(20, dtype=int)

        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_intrinsic_metrics(embeddings, labels)

    def test_all_noise_raises(self):
        embeddings = np.random.randn(20, 2)
        labels = np.full(20, -1, dtype=int)

        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_intrinsic_metrics(embeddings, labels)

    def test_three_clusters(self):
        rng = np.random.RandomState(42)
        c0 = rng.randn(20, 2) + np.array([10.0, 0.0])
        c1 = rng.randn(20, 2) + np.array([0.0, 10.0])
        c2 = rng.randn(20, 2) + np.array([-10.0, 0.0])
        embeddings = np.vstack([c0, c1, c2])
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

        metrics = compute_intrinsic_metrics(embeddings, labels)
        assert metrics.silhouette > 0.7
        assert metrics.calinski_harabasz > 0
        assert metrics.davies_bouldin >= 0


# ------------------------------------------------------------------
# Purity tests
# ------------------------------------------------------------------


class TestPurity:
    """Tests for _compute_purity."""

    def test_perfect_purity(self):
        pred = np.array([0, 0, 0, 1, 1, 1])
        true = np.array([0, 0, 0, 1, 1, 1])
        assert _compute_purity(pred, true) == 1.0

    def test_worst_case_purity(self):
        # Each cluster has equal split of two classes
        pred = np.array([0, 0, 1, 1])
        true = np.array([0, 1, 0, 1])
        # Each cluster picks majority = 1 out of 2, total = 2/4 = 0.5
        assert _compute_purity(pred, true) == 0.5

    def test_partial_purity(self):
        pred = np.array([0, 0, 0, 1, 1, 1])
        true = np.array([0, 0, 1, 1, 1, 0])
        # Cluster 0: [0, 0, 1] → majority 0, count 2
        # Cluster 1: [1, 1, 0] → majority 1, count 2
        # Purity = 4/6
        assert abs(_compute_purity(pred, true) - 4.0 / 6.0) < 1e-10

    def test_empty_arrays(self):
        pred = np.array([], dtype=int)
        true = np.array([], dtype=int)
        assert _compute_purity(pred, true) == 0.0


# ------------------------------------------------------------------
# Extrinsic metrics tests
# ------------------------------------------------------------------


class TestExtrinsicMetrics:
    """Tests for compute_extrinsic_metrics."""

    def test_perfect_clustering(self):
        pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        metrics = compute_extrinsic_metrics(pred, true)
        assert isinstance(metrics, ExtrinsicMetrics)
        assert abs(metrics.nmi - 1.0) < 1e-10
        assert abs(metrics.ari - 1.0) < 1e-10
        assert abs(metrics.purity - 1.0) < 1e-10
        assert abs(metrics.v_measure - 1.0) < 1e-10

    def test_random_clustering_low_scores(self):
        rng = np.random.RandomState(42)
        true = np.array([0] * 50 + [1] * 50 + [2] * 50)
        pred = rng.randint(0, 3, size=150)

        metrics = compute_extrinsic_metrics(pred, true)
        # Random clustering should have low NMI and ARI
        assert metrics.nmi < 0.3
        assert metrics.ari < 0.3

    def test_noise_excluded(self):
        pred = np.array([-1, 0, 0, 0, 1, 1, 1])
        true = np.array([0, 0, 0, 0, 1, 1, 1])

        metrics = compute_extrinsic_metrics(pred, true)
        # After excluding noise (first element), perfect clustering
        assert abs(metrics.nmi - 1.0) < 1e-10
        assert abs(metrics.ari - 1.0) < 1e-10
        assert abs(metrics.purity - 1.0) < 1e-10

    def test_length_mismatch_raises(self):
        pred = np.array([0, 1, 2])
        true = np.array([0, 1])

        with pytest.raises(ValueError, match="same length"):
            compute_extrinsic_metrics(pred, true)

    def test_all_noise_raises(self):
        pred = np.array([-1, -1, -1])
        true = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="No non-noise samples"):
            compute_extrinsic_metrics(pred, true)

    def test_metrics_in_valid_ranges(self):
        pred = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
        true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        metrics = compute_extrinsic_metrics(pred, true)
        assert 0.0 <= metrics.nmi <= 1.0
        assert -1.0 <= metrics.ari <= 1.0
        assert 0.0 <= metrics.purity <= 1.0
        assert 0.0 <= metrics.v_measure <= 1.0


# ------------------------------------------------------------------
# High-level evaluate_clustering tests
# ------------------------------------------------------------------


class TestEvaluateClustering:
    """Tests for the evaluate_clustering function."""

    def test_intrinsic_only(self):
        embeddings, labels = _well_separated_data()
        cr = _make_cluster_result(embeddings, labels)

        result = evaluate_clustering(cr)
        assert isinstance(result, ClusterEvaluation)
        assert result.intrinsic is not None
        assert result.extrinsic is None
        assert result.intrinsic.silhouette > 0.8

    def test_intrinsic_and_extrinsic(self):
        embeddings, labels = _well_separated_data()
        cr = _make_cluster_result(embeddings, labels)
        gold = np.array([0] * 30 + [1] * 30)

        result = evaluate_clustering(cr, gold_labels=gold)
        assert result.intrinsic is not None
        assert result.extrinsic is not None
        assert result.extrinsic.nmi > 0.9
        assert result.extrinsic.purity > 0.9

    def test_single_cluster_intrinsic_none(self):
        embeddings = np.random.randn(20, 2)
        labels = np.zeros(20, dtype=int)
        cr = _make_cluster_result(embeddings, labels)

        result = evaluate_clustering(cr)
        # Single cluster → intrinsic metrics can't be computed
        assert result.intrinsic is None

    def test_with_gold_labels_and_noise(self):
        embeddings, labels = _well_separated_data()
        # Add some noise labels
        labels_noisy = labels.copy()
        labels_noisy[:3] = -1
        cr = _make_cluster_result(embeddings, labels_noisy)
        gold = np.array([0] * 30 + [1] * 30)

        result = evaluate_clustering(cr, gold_labels=gold)
        assert result.intrinsic is not None
        assert result.extrinsic is not None
