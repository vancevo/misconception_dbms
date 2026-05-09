"""Cluster evaluation metrics for misconception mining.

Provides intrinsic and extrinsic metrics for evaluating clustering quality:

Intrinsic (no ground truth needed):
  - Silhouette score
  - Calinski-Harabasz index
  - Davies-Bouldin index

Extrinsic (against gold misconception_tags from Data_Generate):
  - Normalized Mutual Information (NMI)
  - Adjusted Rand Index (ARI)
  - Purity
  - V-measure
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from src.misconception.clustering import ClusterResult


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass
class IntrinsicMetrics:
    """Intrinsic clustering quality metrics.

    Attributes:
        silhouette: Mean silhouette coefficient in [-1, 1].
        calinski_harabasz: Calinski-Harabasz index (higher is better).
        davies_bouldin: Davies-Bouldin index (lower is better).
    """

    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


@dataclass
class ExtrinsicMetrics:
    """Extrinsic clustering quality metrics against gold labels.

    Attributes:
        nmi: Normalized Mutual Information in [0, 1].
        ari: Adjusted Rand Index in [-1, 1].
        purity: Cluster purity in [0, 1].
        v_measure: V-measure (harmonic mean of homogeneity and completeness) in [0, 1].
    """

    nmi: float
    ari: float
    purity: float
    v_measure: float


@dataclass
class ClusterEvaluation:
    """Combined evaluation result for a clustering run.

    Attributes:
        intrinsic: Intrinsic metrics (None if not computable).
        extrinsic: Extrinsic metrics (None if no gold labels provided).
    """

    intrinsic: IntrinsicMetrics | None = None
    extrinsic: ExtrinsicMetrics | None = None


# ------------------------------------------------------------------
# Intrinsic metrics
# ------------------------------------------------------------------


def compute_intrinsic_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> IntrinsicMetrics:
    """Compute intrinsic clustering quality metrics.

    Requires at least 2 clusters and that not all samples belong to
    the same cluster. Noise labels (-1) are excluded before computation.

    Args:
        embeddings: 2-D array of shape (n_samples, dim).
        labels: Cluster label for each sample (-1 = noise).

    Returns:
        An :class:`IntrinsicMetrics` instance.

    Raises:
        ValueError: If fewer than 2 non-noise clusters exist or all
            non-noise samples share a single cluster.
    """
    # Exclude noise points
    mask = labels != -1
    clean_embeddings = embeddings[mask]
    clean_labels = labels[mask]

    unique_labels = set(clean_labels)
    if len(unique_labels) < 2:
        raise ValueError(
            "Intrinsic metrics require at least 2 clusters "
            f"(found {len(unique_labels)} after excluding noise)."
        )

    if len(clean_labels) < 2:
        raise ValueError(
            "Intrinsic metrics require at least 2 non-noise samples."
        )

    sil = silhouette_score(clean_embeddings, clean_labels)
    ch = calinski_harabasz_score(clean_embeddings, clean_labels)
    db = davies_bouldin_score(clean_embeddings, clean_labels)

    return IntrinsicMetrics(
        silhouette=float(sil),
        calinski_harabasz=float(ch),
        davies_bouldin=float(db),
    )


# ------------------------------------------------------------------
# Extrinsic metrics
# ------------------------------------------------------------------


def _compute_purity(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Compute cluster purity.

    Purity is the fraction of samples that are correctly assigned to
    the majority true class within each predicted cluster.

    Args:
        labels_pred: Predicted cluster labels.
        labels_true: Gold/true labels.

    Returns:
        Purity score in [0, 1].
    """
    n = len(labels_true)
    if n == 0:
        return 0.0

    unique_clusters = set(labels_pred)
    total_correct = 0

    for cluster_id in unique_clusters:
        mask = labels_pred == cluster_id
        cluster_true = labels_true[mask]
        if len(cluster_true) == 0:
            continue
        # Count occurrences of each true label in this cluster
        _, counts = np.unique(cluster_true, return_counts=True)
        total_correct += counts.max()

    return total_correct / n


def compute_extrinsic_metrics(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
) -> ExtrinsicMetrics:
    """Compute extrinsic clustering quality metrics against gold labels.

    Noise labels (-1) in ``labels_pred`` are excluded: only samples
    with a non-noise predicted cluster are evaluated.

    Args:
        labels_pred: Predicted cluster labels.
        labels_true: Gold/true labels (e.g., encoded misconception_tags).

    Returns:
        An :class:`ExtrinsicMetrics` instance.

    Raises:
        ValueError: If the arrays have different lengths or are empty
            after noise exclusion.
    """
    if len(labels_pred) != len(labels_true):
        raise ValueError(
            f"labels_pred and labels_true must have the same length "
            f"(got {len(labels_pred)} and {len(labels_true)})."
        )

    # Exclude noise points
    mask = labels_pred != -1
    pred_clean = labels_pred[mask]
    true_clean = labels_true[mask]

    if len(pred_clean) == 0:
        raise ValueError("No non-noise samples to evaluate.")

    nmi = normalized_mutual_info_score(true_clean, pred_clean)
    ari = adjusted_rand_score(true_clean, pred_clean)
    purity = _compute_purity(pred_clean, true_clean)
    vm = v_measure_score(true_clean, pred_clean)

    return ExtrinsicMetrics(
        nmi=float(nmi),
        ari=float(ari),
        purity=float(purity),
        v_measure=float(vm),
    )


# ------------------------------------------------------------------
# High-level evaluator
# ------------------------------------------------------------------


def evaluate_clustering(
    cluster_result: ClusterResult,
    gold_labels: np.ndarray | None = None,
) -> ClusterEvaluation:
    """Evaluate a clustering result with intrinsic and optionally extrinsic metrics.

    Args:
        cluster_result: Output from a clustering method.
        gold_labels: Optional gold-standard labels for extrinsic evaluation
            (e.g., encoded misconception_tags). Must have the same length
            as ``cluster_result.labels``.

    Returns:
        A :class:`ClusterEvaluation` with computed metrics.
    """
    evaluation = ClusterEvaluation()

    # Intrinsic metrics
    try:
        evaluation.intrinsic = compute_intrinsic_metrics(
            cluster_result.embeddings,
            cluster_result.labels,
        )
    except ValueError:
        # Not enough clusters or samples for intrinsic metrics
        evaluation.intrinsic = None

    # Extrinsic metrics
    if gold_labels is not None:
        evaluation.extrinsic = compute_extrinsic_metrics(
            cluster_result.labels,
            gold_labels,
        )

    return evaluation
