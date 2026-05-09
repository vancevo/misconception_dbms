"""Clustering pipeline for misconception mining.

Provides three clustering methods:
  - KMeans: baseline, requires k
  - UMAP + HDBSCAN: primary method, UMAP reduces to n_components=5
  - BERTopic-style: SBERT + UMAP + HDBSCAN + c-TF-IDF (full pipeline)

Also provides c-TF-IDF keyword extraction (top-5 per cluster).

All heavy external dependencies (umap-learn, hdbscan) are imported
lazily so the module can be imported in environments where they are
not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class ClusteringMethod(str, Enum):
    """Supported clustering methods."""

    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    BERTOPIC = "bertopic"


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Output of a clustering run.

    Attributes:
        labels: Cluster assignment for each input sample.
            -1 indicates noise (HDBSCAN).
        method: The clustering method used.
        n_clusters: Number of clusters found (excluding noise).
        embeddings: The (possibly reduced) embeddings used.
        keywords: Per-cluster top keywords (cluster_id → list).
    """

    labels: np.ndarray
    method: ClusteringMethod
    n_clusters: int
    embeddings: np.ndarray
    keywords: dict[int, list[str]] = field(default_factory=dict)


# ------------------------------------------------------------------
# Lazy imports
# ------------------------------------------------------------------

def _import_umap():
    """Lazily import UMAP."""
    try:
        import umap  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP dimensionality reduction. "
            "Install it with: pip install umap-learn"
        ) from exc
    return umap


def _import_hdbscan():
    """Lazily import HDBSCAN."""
    try:
        import hdbscan  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required for HDBSCAN clustering. "
            "Install it with: pip install hdbscan"
        ) from exc
    return hdbscan


# ------------------------------------------------------------------
# KMeans clustering
# ------------------------------------------------------------------

def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int | None = 42,
) -> ClusterResult:
    """Run KMeans clustering on *embeddings*.

    Args:
        embeddings: 2-D array of shape (n_samples, dim).
        n_clusters: Number of clusters.
        n_init: Number of KMeans initialisations.
        max_iter: Maximum iterations per run.
        random_state: Seed for reproducibility.

    Returns:
        A :class:`ClusterResult` with cluster labels.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415

    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = km.fit_predict(embeddings)
    return ClusterResult(
        labels=labels,
        method=ClusteringMethod.KMEANS,
        n_clusters=n_clusters,
        embeddings=embeddings,
    )


# ------------------------------------------------------------------
# UMAP dimensionality reduction
# ------------------------------------------------------------------

def reduce_umap(
    embeddings: np.ndarray,
    *,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int | None = 42,
) -> np.ndarray:
    """Reduce *embeddings* with UMAP.

    Args:
        embeddings: 2-D array of shape (n_samples, dim).
        n_components: Target dimensionality.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        metric: Distance metric.
        random_state: Seed for reproducibility.

    Returns:
        Reduced embeddings of shape (n_samples, n_components).
    """
    umap = _import_umap()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# ------------------------------------------------------------------
# HDBSCAN clustering
# ------------------------------------------------------------------

def cluster_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_method: str = "eom",
) -> ClusterResult:
    """Run HDBSCAN on *embeddings*.

    Args:
        embeddings: 2-D array (n_samples, dim).
        min_cluster_size: Minimum cluster size.
        min_samples: Minimum samples for core point.
        cluster_selection_method: ``"eom"`` or ``"leaf"``.

    Returns:
        A :class:`ClusterResult` with cluster labels (-1 = noise).
    """
    hdbscan_mod = _import_hdbscan()
    clusterer = hdbscan_mod.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels) - {-1})
    return ClusterResult(
        labels=labels,
        method=ClusteringMethod.HDBSCAN,
        n_clusters=n_clusters,
        embeddings=embeddings,
    )


# ------------------------------------------------------------------
# UMAP + HDBSCAN combined
# ------------------------------------------------------------------

def cluster_umap_hdbscan(
    embeddings: np.ndarray,
    *,
    # UMAP params
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    umap_metric: str = "cosine",
    random_state: int | None = 42,
    # HDBSCAN params
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_method: str = "eom",
) -> ClusterResult:
    """UMAP dimensionality reduction followed by HDBSCAN clustering.

    Args:
        embeddings: 2-D array (n_samples, dim).
        n_components: UMAP target dimensionality (default 5).
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        umap_metric: UMAP distance metric.
        random_state: UMAP random seed.
        min_cluster_size: HDBSCAN min_cluster_size.
        min_samples: HDBSCAN min_samples.
        cluster_selection_method: HDBSCAN selection method.

    Returns:
        A :class:`ClusterResult` with reduced embeddings stored.
    """
    reduced = reduce_umap(
        embeddings,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    result = cluster_hdbscan(
        reduced,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    # Store the reduced embeddings in the result
    result.embeddings = reduced
    return result


# ------------------------------------------------------------------
# c-TF-IDF keyword extraction
# ------------------------------------------------------------------

def extract_ctfidf_keywords(
    documents: Sequence[str],
    labels: np.ndarray,
    *,
    top_n: int = 5,
) -> dict[int, list[str]]:
    """Extract top-*top_n* keywords per cluster using c-TF-IDF.

    c-TF-IDF concatenates all documents in a cluster into a single
    "class document", then applies TF-IDF across class documents to
    find the most representative terms per cluster.

    Noise samples (label == -1) are excluded.

    Args:
        documents: The original text for each sample.
        labels: Cluster label for each sample.
        top_n: Number of keywords to extract per cluster.

    Returns:
        Mapping from cluster id to a list of top keywords.
    """
    from sklearn.feature_extraction.text import (  # noqa: PLC0415
        TfidfVectorizer,
    )

    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        return {}

    # Build one "class document" per cluster
    class_docs: list[str] = []
    cluster_ids: list[int] = []
    for cid in unique_labels:
        mask = labels == cid
        merged = " ".join(
            doc for doc, m in zip(documents, mask) if m
        )
        class_docs.append(merged)
        cluster_ids.append(cid)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(class_docs)
    feature_names = vectorizer.get_feature_names_out()

    keywords: dict[int, list[str]] = {}
    for idx, cid in enumerate(cluster_ids):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords[cid] = [feature_names[i] for i in top_indices]

    return keywords


# ------------------------------------------------------------------
# BERTopic-style pipeline
# ------------------------------------------------------------------

def cluster_bertopic(
    embeddings: np.ndarray,
    documents: Sequence[str],
    *,
    # UMAP params
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    umap_metric: str = "cosine",
    random_state: int | None = 42,
    # HDBSCAN params
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_method: str = "eom",
    # c-TF-IDF params
    top_n_keywords: int = 5,
) -> ClusterResult:
    """Full BERTopic-style pipeline: SBERT + UMAP + HDBSCAN + c-TF-IDF.

    Assumes *embeddings* are already produced by an SBERT model.
    Applies UMAP reduction, HDBSCAN clustering, then c-TF-IDF
    keyword extraction.

    Args:
        embeddings: 2-D SBERT embeddings (n_samples, dim).
        documents: Original text for each sample.
        n_components: UMAP target dimensionality.
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
        umap_metric: UMAP distance metric.
        random_state: UMAP random seed.
        min_cluster_size: HDBSCAN min_cluster_size.
        min_samples: HDBSCAN min_samples.
        cluster_selection_method: HDBSCAN selection method.
        top_n_keywords: Number of c-TF-IDF keywords per cluster.

    Returns:
        A :class:`ClusterResult` with keywords populated.
    """
    result = cluster_umap_hdbscan(
        embeddings,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        umap_metric=umap_metric,
        random_state=random_state,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )
    result.method = ClusteringMethod.BERTOPIC
    result.keywords = extract_ctfidf_keywords(
        documents, result.labels, top_n=top_n_keywords
    )
    return result
