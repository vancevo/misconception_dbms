"""Phase 3: Misconception Mining Experiments

Runs all embedding strategies × clustering methods on Data_Generate
incorrect answers. Supports:
  - 3 embedding strategies (A: answer-only, B: question+answer, C: full triplet)
  - 3 clustering methods (KMeans, UMAP+HDBSCAN, BERTopic)
  - 3 granularity levels (per-question, per-domain, global)
  - Extrinsic validation against gold misconception_tags (NMI, ARI, Purity, V-measure)

All results are saved to results/phase3/ as JSON.

Usage:
    python experiments/phase3_misconception.py
    python experiments/phase3_misconception.py --config configs/misconception.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import UnifiedRecord
from src.evaluation.reporting import save_results
from src.misconception.clustering import (
    ClusteringMethod,
    cluster_bertopic,
    cluster_kmeans,
    cluster_umap_hdbscan,
    extract_ctfidf_keywords,
)
from src.misconception.embedder import (
    EmbeddingResult,
    EmbeddingStrategy,
    Granularity,
    MisconceptionEmbedder,
    filter_misconception_records,
)
from src.misconception.evaluator import (
    evaluate_clustering,
    compute_extrinsic_metrics,
    compute_intrinsic_metrics,
)
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3_misconception")

CONFIG_PATH = PROJECT_ROOT / "configs" / "misconception.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_unified_records(unified_dir: Path) -> list[UnifiedRecord]:
    """Load all unified JSONL files into UnifiedRecord instances."""
    all_records: list[UnifiedRecord] = []
    for jsonl_file in sorted(unified_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping malformed JSON in %s line %d: %s",
                        jsonl_file.name, line_num, e,
                    )
                    continue
                valid_fields = {
                    fld.name for fld in dataclasses.fields(UnifiedRecord)
                }
                filtered = {k: v for k, v in data.items() if k in valid_fields}
                try:
                    rec = UnifiedRecord(**filtered)
                    all_records.append(rec)
                except (TypeError, ValueError) as e:
                    logger.warning("Skipping record: %s", e)
    return all_records


# ---------------------------------------------------------------------------
# Gold label encoding
# ---------------------------------------------------------------------------

def encode_gold_labels(records: list[UnifiedRecord]) -> np.ndarray | None:
    """Encode misconception_tags into integer labels for extrinsic evaluation.

    Uses the first misconception tag as the gold label. Records without
    misconception_tags are assigned label -1 (treated as noise).

    Returns None if no records have misconception_tags.
    """
    tag_to_id: dict[str, int] = {}
    labels: list[int] = []

    for rec in records:
        if rec.misconception_tags:
            tag = rec.misconception_tags[0]
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
            labels.append(tag_to_id[tag])
        else:
            labels.append(-1)

    if not tag_to_id:
        return None

    return np.array(labels)


# ---------------------------------------------------------------------------
# Prepare text for c-TF-IDF (needed by BERTopic and keyword extraction)
# ---------------------------------------------------------------------------

def _prepare_text(record: UnifiedRecord, strategy: EmbeddingStrategy) -> str:
    """Build the text string for a given strategy (mirrors embedder logic)."""
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
# Clustering dispatcher
# ---------------------------------------------------------------------------

STRATEGY_MAP: dict[str, EmbeddingStrategy] = {
    "answer_only": EmbeddingStrategy.ANSWER_ONLY,
    "question_answer": EmbeddingStrategy.QUESTION_ANSWER,
    "full_triplet": EmbeddingStrategy.FULL_TRIPLET,
}

GRANULARITY_MAP: dict[str, Granularity] = {
    "per_question": Granularity.PER_QUESTION,
    "per_domain": Granularity.PER_DOMAIN,
    "global": Granularity.GLOBAL,
}


def run_clustering(
    embedding_result: EmbeddingResult,
    method: str,
    cfg: dict,
) -> dict[str, Any]:
    """Run a single clustering method on an EmbeddingResult.

    Args:
        embedding_result: Output from MisconceptionEmbedder.embed().
        method: One of "kmeans", "umap_hdbscan", "bertopic".
        cfg: Full config dict.

    Returns:
        Dict with clustering results, metrics, and keywords.
    """
    embeddings = embedding_result.embeddings
    records = embedding_result.records
    documents = [_prepare_text(r, embedding_result.strategy) for r in records]

    # Extract config sections
    kmeans_cfg = cfg.get("kmeans", {})
    umap_cfg = cfg.get("umap", {})
    hdbscan_cfg = cfg.get("hdbscan", {})
    ctfidf_cfg = cfg.get("ctfidf", {})
    seed = cfg.get("seed", 42)

    result_dict: dict[str, Any] = {
        "method": method,
        "strategy": embedding_result.strategy.value,
        "granularity": embedding_result.granularity.value,
        "group_key": embedding_result.group_key,
        "n_samples": len(records),
    }

    try:
        if method == "kmeans":
            n_clusters = min(kmeans_cfg.get("n_clusters", 10), len(records))
            if n_clusters < 2:
                result_dict["error"] = "Too few samples for KMeans"
                return result_dict
            cluster_result = cluster_kmeans(
                embeddings,
                n_clusters=n_clusters,
                n_init=kmeans_cfg.get("n_init", 10),
                max_iter=kmeans_cfg.get("max_iter", 300),
                random_state=seed,
            )
            # Extract keywords via c-TF-IDF
            cluster_result.keywords = extract_ctfidf_keywords(
                documents,
                cluster_result.labels,
                top_n=ctfidf_cfg.get("top_n_keywords", 5),
            )

        elif method == "umap_hdbscan":
            if len(records) < umap_cfg.get("n_neighbors", 15):
                result_dict["error"] = "Too few samples for UMAP+HDBSCAN"
                return result_dict
            cluster_result = cluster_umap_hdbscan(
                embeddings,
                n_components=umap_cfg.get("n_components", 5),
                n_neighbors=umap_cfg.get("n_neighbors", 15),
                min_dist=umap_cfg.get("min_dist", 0.1),
                umap_metric=umap_cfg.get("metric", "cosine"),
                random_state=seed,
                min_cluster_size=hdbscan_cfg.get("min_cluster_size", 5),
                min_samples=hdbscan_cfg.get("min_samples", 3),
                cluster_selection_method=hdbscan_cfg.get(
                    "cluster_selection_method", "eom"
                ),
            )
            cluster_result.keywords = extract_ctfidf_keywords(
                documents,
                cluster_result.labels,
                top_n=ctfidf_cfg.get("top_n_keywords", 5),
            )

        elif method == "bertopic":
            if len(records) < umap_cfg.get("n_neighbors", 15):
                result_dict["error"] = "Too few samples for BERTopic"
                return result_dict
            cluster_result = cluster_bertopic(
                embeddings,
                documents,
                n_components=umap_cfg.get("n_components", 5),
                n_neighbors=umap_cfg.get("n_neighbors", 15),
                min_dist=umap_cfg.get("min_dist", 0.1),
                umap_metric=umap_cfg.get("metric", "cosine"),
                random_state=seed,
                min_cluster_size=hdbscan_cfg.get("min_cluster_size", 5),
                min_samples=hdbscan_cfg.get("min_samples", 3),
                cluster_selection_method=hdbscan_cfg.get(
                    "cluster_selection_method", "eom"
                ),
                top_n_keywords=ctfidf_cfg.get("top_n_keywords", 5),
            )
        else:
            result_dict["error"] = f"Unknown clustering method: {method}"
            return result_dict

    except Exception as e:
        result_dict["error"] = str(e)
        logger.warning(
            "Clustering failed (method=%s, strategy=%s, group=%s): %s",
            method, embedding_result.strategy.value,
            embedding_result.group_key, e,
        )
        return result_dict

    result_dict["n_clusters"] = cluster_result.n_clusters
    result_dict["keywords"] = {
        str(k): v for k, v in cluster_result.keywords.items()
    }

    # Intrinsic metrics
    try:
        intrinsic = compute_intrinsic_metrics(
            cluster_result.embeddings, cluster_result.labels
        )
        result_dict["intrinsic"] = {
            "silhouette": intrinsic.silhouette,
            "calinski_harabasz": intrinsic.calinski_harabasz,
            "davies_bouldin": intrinsic.davies_bouldin,
        }
    except ValueError as e:
        result_dict["intrinsic"] = {"error": str(e)}

    # Extrinsic metrics against gold misconception_tags
    gold_labels = encode_gold_labels(records)
    if gold_labels is not None:
        try:
            extrinsic = compute_extrinsic_metrics(
                cluster_result.labels, gold_labels
            )
            result_dict["extrinsic"] = {
                "nmi": extrinsic.nmi,
                "ari": extrinsic.ari,
                "purity": extrinsic.purity,
                "v_measure": extrinsic.v_measure,
            }
        except ValueError as e:
            result_dict["extrinsic"] = {"error": str(e)}
    else:
        result_dict["extrinsic"] = {"error": "No gold misconception_tags available"}

    return result_dict


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

CLUSTERING_METHODS = ["kmeans", "umap_hdbscan", "bertopic"]


def run_global_experiments(
    records: list[UnifiedRecord],
    embedder: MisconceptionEmbedder,
    cfg: dict,
) -> list[dict[str, Any]]:
    """Run all embedding strategies × all clustering methods at global granularity.

    This is the primary experiment grid (sub-task 20.2).
    """
    logger.info("=" * 70)
    logger.info("  GLOBAL EXPERIMENTS: 3 strategies × 3 clustering methods")
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []
    strategies = cfg.get("embedding_strategies", [])

    for strat_cfg in strategies:
        strat_name = strat_cfg["name"]
        strategy = STRATEGY_MAP.get(strat_name)
        if strategy is None:
            logger.warning("Unknown strategy: %s; skipping", strat_name)
            continue

        logger.info("-" * 50)
        logger.info("  Strategy: %s", strat_name)
        logger.info("-" * 50)

        # Embed at global granularity
        embedding_results = embedder.embed(
            records, strategy, Granularity.GLOBAL
        )

        if not embedding_results:
            logger.warning("  No embedding results for strategy %s", strat_name)
            continue

        for emb_result in embedding_results:
            for method in CLUSTERING_METHODS:
                logger.info(
                    "  Running %s on %s (group=%s, n=%d)",
                    method, strat_name, emb_result.group_key,
                    len(emb_result.records),
                )
                result = run_clustering(emb_result, method, cfg)
                result["experiment"] = "global_grid"
                all_results.append(result)

    return all_results


def run_granularity_experiments(
    records: list[UnifiedRecord],
    embedder: MisconceptionEmbedder,
    cfg: dict,
) -> list[dict[str, Any]]:
    """Run per-question and per-domain granularity experiments (sub-task 20.3).

    Uses the first embedding strategy from config for each granularity level.
    Runs all clustering methods on each group.
    """
    logger.info("=" * 70)
    logger.info("  GRANULARITY EXPERIMENTS: per-question and per-domain")
    logger.info("=" * 70)

    all_results: list[dict[str, Any]] = []
    strategies = cfg.get("embedding_strategies", [])
    granularity_levels = cfg.get("granularity_levels", [])

    # Filter to per_question and per_domain only
    target_granularities = [
        g for g in granularity_levels if g in ("per_question", "per_domain")
    ]

    for strat_cfg in strategies:
        strat_name = strat_cfg["name"]
        strategy = STRATEGY_MAP.get(strat_name)
        if strategy is None:
            continue

        for gran_name in target_granularities:
            granularity = GRANULARITY_MAP.get(gran_name)
            if granularity is None:
                continue

            logger.info("-" * 50)
            logger.info(
                "  Strategy: %s, Granularity: %s", strat_name, gran_name
            )
            logger.info("-" * 50)

            embedding_results = embedder.embed(records, strategy, granularity)

            if not embedding_results:
                logger.warning(
                    "  No embedding results for %s/%s", strat_name, gran_name
                )
                continue

            logger.info(
                "  Got %d groups for %s/%s",
                len(embedding_results), strat_name, gran_name,
            )

            for emb_result in embedding_results:
                for method in CLUSTERING_METHODS:
                    logger.info(
                        "    Running %s on group=%s (n=%d)",
                        method, emb_result.group_key, len(emb_result.records),
                    )
                    result = run_clustering(emb_result, method, cfg)
                    result["experiment"] = "granularity"
                    all_results.append(result)

    return all_results


def build_summary(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a summary of all experiment results.

    Extracts the best-performing strategy × method combinations
    by NMI and ARI from the global grid experiments.
    """
    summary: dict[str, Any] = {
        "total_experiments": len(all_results),
        "successful": sum(1 for r in all_results if "error" not in r),
        "failed": sum(1 for r in all_results if "error" in r),
    }

    # Find best global grid result by NMI
    global_results = [
        r for r in all_results
        if r.get("experiment") == "global_grid"
        and "extrinsic" in r
        and isinstance(r["extrinsic"], dict)
        and "nmi" in r["extrinsic"]
    ]

    if global_results:
        best_nmi = max(global_results, key=lambda r: r["extrinsic"]["nmi"])
        summary["best_by_nmi"] = {
            "strategy": best_nmi.get("strategy"),
            "method": best_nmi.get("method"),
            "nmi": best_nmi["extrinsic"]["nmi"],
            "ari": best_nmi["extrinsic"].get("ari"),
            "purity": best_nmi["extrinsic"].get("purity"),
            "v_measure": best_nmi["extrinsic"].get("v_measure"),
        }

        best_ari = max(global_results, key=lambda r: r["extrinsic"]["ari"])
        summary["best_by_ari"] = {
            "strategy": best_ari.get("strategy"),
            "method": best_ari.get("method"),
            "ari": best_ari["extrinsic"]["ari"],
            "nmi": best_ari["extrinsic"].get("nmi"),
        }

    # Keyword summaries from global grid
    keyword_summaries: list[dict[str, Any]] = []
    for r in global_results:
        if r.get("keywords"):
            keyword_summaries.append({
                "strategy": r.get("strategy"),
                "method": r.get("method"),
                "n_clusters": r.get("n_clusters"),
                "keywords": r.get("keywords"),
            })
    summary["keyword_summaries"] = keyword_summaries

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 3 misconception mining experiments."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Misconception Mining Experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to misconception config YAML (default: configs/misconception.yaml)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["global", "granularity"],
        default=["global", "granularity"],
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info("Loaded config from %s (seed=%d)", config_path, seed)

    # Load unified data
    unified_dir = PROJECT_ROOT / "data" / "unified"
    logger.info("Loading unified records from %s ...", unified_dir)
    all_records = load_unified_records(unified_dir)
    logger.info("Loaded %d total records", len(all_records))

    if not all_records:
        logger.error("No records loaded. Run phase1_data_audit.py first.")
        sys.exit(1)

    # Filter to misconception-relevant records
    misconception_records = filter_misconception_records(all_records)
    logger.info(
        "Filtered to %d misconception-relevant records (label_5way ∈ "
        "{partially_correct_incomplete, contradictory, irrelevant})",
        len(misconception_records),
    )

    if not misconception_records:
        logger.error("No misconception-relevant records found.")
        sys.exit(1)

    # Create embedder
    sbert_model = cfg.get("sbert_model", "all-MiniLM-L6-v2")
    embedder = MisconceptionEmbedder(model_name=sbert_model)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_experiment_results: dict[str, Any] = {}

    # --- Global grid experiments (20.2) ---
    if "global" in args.experiments:
        global_results = run_global_experiments(
            misconception_records, embedder, cfg
        )
        all_experiment_results["global_grid"] = global_results
        save_results(
            {"experiments": global_results},
            "global_grid_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info(
            "Saved global grid results (%d experiments)", len(global_results)
        )

    # --- Granularity experiments (20.3) ---
    if "granularity" in args.experiments:
        granularity_results = run_granularity_experiments(
            misconception_records, embedder, cfg
        )
        all_experiment_results["granularity"] = granularity_results
        save_results(
            {"experiments": granularity_results},
            "granularity_results",
            results_dir=str(RESULTS_DIR),
        )
        logger.info(
            "Saved granularity results (%d experiments)",
            len(granularity_results),
        )

    # --- Build summary with keyword summaries (20.4, 20.5) ---
    all_flat = []
    for v in all_experiment_results.values():
        if isinstance(v, list):
            all_flat.extend(v)
    summary = build_summary(all_flat)
    all_experiment_results["summary"] = summary

    # Save combined results
    save_results(
        all_experiment_results,
        "phase3_all_results",
        results_dir=str(RESULTS_DIR),
    )

    # Save keyword summaries separately
    save_results(
        {"keyword_summaries": summary.get("keyword_summaries", [])},
        "cluster_keyword_summaries",
        results_dir=str(RESULTS_DIR),
    )

    logger.info("=" * 70)
    logger.info("  Phase 3 misconception mining experiments complete.")
    logger.info("  Results saved to: %s", RESULTS_DIR)
    logger.info("  Total experiments: %d", summary["total_experiments"])
    logger.info("  Successful: %d", summary["successful"])
    logger.info("  Failed: %d", summary["failed"])
    if "best_by_nmi" in summary:
        best = summary["best_by_nmi"]
        logger.info(
            "  Best by NMI: %s + %s (NMI=%.4f, ARI=%.4f)",
            best["strategy"], best["method"],
            best["nmi"], best.get("ari", 0),
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
