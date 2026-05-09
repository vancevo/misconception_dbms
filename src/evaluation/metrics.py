"""Shared evaluation harness for classification, regression, and feedback metrics."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
)
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_rel, chi2

from src.evaluation.bootstrap import bootstrap_ci


class EvaluationHarness:
    """Computes classification and regression metrics with bootstrap CIs."""

    def classification_metrics(
        self,
        y_true: list[str],
        y_pred: list[str],
        bootstrap_n: int = 1000,
    ) -> dict:
        """Compute classification metrics with 95% bootstrap confidence intervals.

        Returns:
            dict with keys: accuracy, macro_f1, weighted_f1, per_class_f1,
            confusion_matrix — each (except confusion_matrix) has 'value', 'ci_lower', 'ci_upper'.
        """
        y_true = list(y_true)
        y_pred = list(y_pred)

        classes = sorted(set(y_true) | set(y_pred))

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        per_class = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

        # Bootstrap CIs
        acc_ci = bootstrap_ci(
            y_true, y_pred,
            lambda yt, yp: accuracy_score(yt, yp),
            n=bootstrap_n,
        )
        macro_f1_ci = bootstrap_ci(
            y_true, y_pred,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            n=bootstrap_n,
        )
        weighted_f1_ci = bootstrap_ci(
            y_true, y_pred,
            lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
            n=bootstrap_n,
        )

        per_class_f1_with_ci = {}
        for i, cls in enumerate(classes):
            cls_ci = bootstrap_ci(
                y_true, y_pred,
                lambda yt, yp, c=cls: f1_score(yt, yp, average=None, labels=[c], zero_division=0)[0],
                n=bootstrap_n,
            )
            per_class_f1_with_ci[cls] = {
                "value": float(per_class[i]),
                "ci_lower": cls_ci[0],
                "ci_upper": cls_ci[1],
            }

        return {
            "accuracy": {
                "value": float(acc),
                "ci_lower": acc_ci[0],
                "ci_upper": acc_ci[1],
            },
            "macro_f1": {
                "value": float(macro_f1),
                "ci_lower": macro_f1_ci[0],
                "ci_upper": macro_f1_ci[1],
            },
            "weighted_f1": {
                "value": float(weighted_f1),
                "ci_lower": weighted_f1_ci[0],
                "ci_upper": weighted_f1_ci[1],
            },
            "per_class_f1": per_class_f1_with_ci,
            "confusion_matrix": {
                "labels": classes,
                "matrix": cm,
            },
        }

    def regression_metrics(
        self,
        y_true: list[float],
        y_pred: list[float],
        bootstrap_n: int = 1000,
    ) -> dict:
        """Compute regression metrics with 95% bootstrap confidence intervals.

        Returns:
            dict with keys: pearson_r, spearman_rho, rmse, mae, qwk —
            each has 'value', 'ci_lower', 'ci_upper'.
        """
        y_true = list(y_true)
        y_pred = list(y_pred)

        def _pearson(yt, yp):
            r, _ = pearsonr(yt, yp)
            return float(r)

        def _spearman(yt, yp):
            rho, _ = spearmanr(yt, yp)
            return float(rho)

        def _rmse(yt, yp):
            return float(np.sqrt(np.mean((np.array(yt) - np.array(yp)) ** 2)))

        def _mae(yt, yp):
            return float(np.mean(np.abs(np.array(yt) - np.array(yp))))

        def _qwk(yt, yp):
            # QWK requires integer/ordinal labels; round to nearest int
            yt_int = [round(v) for v in yt]
            yp_int = [round(v) for v in yp]
            return float(cohen_kappa_score(yt_int, yp_int, weights="quadratic"))

        metrics_fns = {
            "pearson_r": _pearson,
            "spearman_rho": _spearman,
            "rmse": _rmse,
            "mae": _mae,
            "qwk": _qwk,
        }

        result = {}
        for name, fn in metrics_fns.items():
            value = fn(y_true, y_pred)
            ci = bootstrap_ci(y_true, y_pred, fn, n=bootstrap_n)
            result[name] = {
                "value": value,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }

        return result

    def compare_models(
        self,
        y_true: list,
        y_pred_a: list,
        y_pred_b: list,
        task: str,
    ) -> dict:
        """Compare two models using statistical tests.

        Args:
            y_true: Ground truth labels/values.
            y_pred_a: Predictions from model A.
            y_pred_b: Predictions from model B.
            task: "classification" or "regression".

        Returns:
            dict with 'mcnemar_p' (classification) or 'paired_t_p' (regression).
        """
        y_true = list(y_true)
        y_pred_a = list(y_pred_a)
        y_pred_b = list(y_pred_b)

        if task == "classification":
            # Build 2x2 contingency table for McNemar's test
            # a_correct[i] = True if model A correct on sample i
            a_correct = [yt == yp for yt, yp in zip(y_true, y_pred_a)]
            b_correct = [yt == yp for yt, yp in zip(y_true, y_pred_b)]

            # Contingency table:
            # [both_correct, a_correct_b_wrong]
            # [a_wrong_b_correct, both_wrong]
            n00 = sum(1 for a, b in zip(a_correct, b_correct) if a and b)
            n01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
            n10 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
            n11 = sum(1 for a, b in zip(a_correct, b_correct) if not a and not b)

            # McNemar's test with continuity correction (chi-squared approximation)
            # Only the discordant cells (n01, n10) matter
            discordant = n01 + n10
            if discordant == 0:
                # No discordant pairs → models are identical → p = 1.0
                return {"mcnemar_p": 1.0}
            # With continuity correction: chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
            chi2_stat = (abs(n01 - n10) - 1) ** 2 / discordant
            p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
            return {"mcnemar_p": float(p_value)}

        elif task == "regression":
            errors_a = [float(yt) - float(yp) for yt, yp in zip(y_true, y_pred_a)]
            errors_b = [float(yt) - float(yp) for yt, yp in zip(y_true, y_pred_b)]
            _, p_value = ttest_rel(errors_a, errors_b)
            return {"paired_t_p": float(p_value)}

        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

    # ------------------------------------------------------------------
    # Feedback evaluation metrics (Requirement 21)
    # ------------------------------------------------------------------

    def feedback_metrics(
        self,
        generated_feedback: list[str],
        gold_feedback: list[str],
        gold_missing_concepts: list[list[str]] | None = None,
        reference_answers: list[str] | None = None,
        *,
        nli_pipeline: Any | None = None,
        bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
    ) -> dict:
        """Compute all feedback evaluation metrics.

        Parameters
        ----------
        generated_feedback : list[str]
            Generated feedback texts.
        gold_feedback : list[str]
            Gold/reference feedback texts.
        gold_missing_concepts : list[list[str]] | None
            Gold missing concepts per record (for concept coverage).
        reference_answers : list[str] | None
            Reference answers per record (for factual consistency
            and hallucination rate).
        nli_pipeline : Any | None
            Injected NLI pipeline for factual consistency /
            hallucination checks. If ``None``, those metrics are
            skipped.
        bertscore_model_type : str
            Model type for BERTScore computation.

        Returns
        -------
        dict
            Keys: ``rouge_l``, ``bertscore``, ``concept_coverage``,
            ``factual_consistency``, ``hallucination_rate``.
        """
        result: dict[str, Any] = {}

        result["rouge_l"] = compute_rouge_l(
            generated_feedback, gold_feedback,
        )
        result["bertscore"] = compute_bertscore(
            generated_feedback, gold_feedback,
            model_type=bertscore_model_type,
        )

        if gold_missing_concepts is not None:
            result["concept_coverage"] = compute_concept_coverage(
                generated_feedback, gold_missing_concepts,
            )

        if reference_answers is not None:
            result["factual_consistency"] = compute_factual_consistency(
                generated_feedback, reference_answers,
                nli_pipeline=nli_pipeline,
            )
            result["hallucination_rate"] = compute_hallucination_rate(
                generated_feedback, reference_answers,
                nli_pipeline=nli_pipeline,
            )

        return result


# ======================================================================
# ROUGE-L computation (Task 26.1)
# ======================================================================

def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute the length of the longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimised DP (two rows).
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_sentence(hypothesis: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 for a single pair."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge_l(
    generated: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute corpus-level ROUGE-L (average F1, precision, recall).

    Parameters
    ----------
    generated : list[str]
        Generated feedback texts.
    references : list[str]
        Gold feedback texts.

    Returns
    -------
    dict
        ``{"precision": ..., "recall": ..., "f1": ...}``
    """
    if not generated or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precisions, recalls, f1s = [], [], []
    for gen, ref in zip(generated, references):
        scores = rouge_l_sentence(gen, ref)
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])
        f1s.append(scores["f1"])

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


# ======================================================================
# BERTScore computation (Task 26.1)
# ======================================================================

def compute_bertscore(
    generated: list[str],
    references: list[str],
    *,
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, float]:
    """Compute corpus-level BERTScore (average precision, recall, F1).

    Attempts to use the ``bert_score`` library. If unavailable, falls
    back to a lightweight cosine-similarity proxy using TF-IDF vectors.

    Parameters
    ----------
    generated : list[str]
        Generated feedback texts.
    references : list[str]
        Gold feedback texts.
    model_type : str
        Model type for the ``bert_score`` library.

    Returns
    -------
    dict
        ``{"precision": ..., "recall": ..., "f1": ...}``
    """
    if not generated or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        from bert_score import score as bert_score_fn  # noqa: WPS433

        P, R, F1 = bert_score_fn(
            generated, references,
            model_type=model_type,
            verbose=False,
        )
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }
    except ImportError:
        # Fallback: TF-IDF cosine similarity as a proxy.
        return _tfidf_similarity_proxy(generated, references)


def _tfidf_similarity_proxy(
    generated: list[str],
    references: list[str],
) -> dict[str, float]:
    """Lightweight BERTScore proxy using TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    all_texts = generated + references
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(all_texts)
    except ValueError:
        # All texts are empty or contain only stop words.
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    n = len(generated)
    gen_vecs = tfidf[:n]
    ref_vecs = tfidf[n:]

    sims = []
    for i in range(n):
        sim = cosine_similarity(gen_vecs[i], ref_vecs[i])[0][0]
        sims.append(float(sim))

    avg = float(np.mean(sims))
    return {"precision": avg, "recall": avg, "f1": avg}


# ======================================================================
# Concept coverage metric (Task 26.2)
# ======================================================================

def concept_coverage_single(
    feedback: str,
    missing_concepts: list[str],
) -> float:
    """Fraction of gold missing_concepts mentioned in feedback.

    A concept is considered "mentioned" if all its tokens appear in
    the feedback text (case-insensitive).
    """
    if not missing_concepts:
        return 1.0  # No concepts to cover → perfect coverage.

    feedback_lower = feedback.lower()
    mentioned = 0
    for concept in missing_concepts:
        concept_tokens = concept.lower().split()
        if all(tok in feedback_lower for tok in concept_tokens):
            mentioned += 1

    return mentioned / len(missing_concepts)


def compute_concept_coverage(
    generated: list[str],
    gold_missing_concepts: list[list[str]],
) -> dict[str, float]:
    """Compute corpus-level concept coverage.

    Parameters
    ----------
    generated : list[str]
        Generated feedback texts.
    gold_missing_concepts : list[list[str]]
        Gold missing concepts per record.

    Returns
    -------
    dict
        ``{"mean": ..., "per_record": [...]}``
    """
    if not generated:
        return {"mean": 0.0, "per_record": []}

    per_record = []
    for fb, concepts in zip(generated, gold_missing_concepts):
        per_record.append(concept_coverage_single(fb, concepts))

    return {
        "mean": float(np.mean(per_record)),
        "per_record": per_record,
    }


# ======================================================================
# Factual consistency metric (Task 26.3)
# ======================================================================

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple heuristic."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# NLI label sets (consistent with concept_gap.py / generative.py)
_ENTAILMENT_LABELS = frozenset({
    "entailment", "ENTAILMENT", "LABEL_0",
})
_NEUTRAL_LABELS = frozenset({
    "neutral", "NEUTRAL", "LABEL_1",
})
_CONTRADICTION_LABELS = frozenset({
    "contradiction", "CONTRADICTION", "LABEL_2",
})


def factual_consistency_single(
    feedback: str,
    reference_answer: str,
    nli_pipeline: Any,
) -> float:
    """Fraction of feedback sentences entailed by or neutral to reference.

    Parameters
    ----------
    feedback : str
        Generated feedback text.
    reference_answer : str
        The reference answer to check against.
    nli_pipeline : Any
        An NLI pipeline (HuggingFace or mock) that accepts
        ``{"text": ..., "text_pair": ...}`` and returns
        ``[{"label": ..., "score": ...}]``.

    Returns
    -------
    float
        Fraction of sentences that are entailed or neutral (not
        contradicting).
    """
    sentences = _split_sentences(feedback)
    if not sentences:
        return 1.0

    consistent_count = 0
    for sentence in sentences:
        result = nli_pipeline(
            {"text": reference_answer, "text_pair": sentence},
            top_k=1,
        )
        if result and isinstance(result, list):
            top = result[0] if isinstance(result[0], dict) else result[0][0]
            label = top["label"]
        else:
            label = "neutral"

        if label not in _CONTRADICTION_LABELS:
            consistent_count += 1

    return consistent_count / len(sentences)


def compute_factual_consistency(
    generated: list[str],
    reference_answers: list[str],
    *,
    nli_pipeline: Any | None = None,
) -> dict[str, float]:
    """Compute corpus-level factual consistency.

    Parameters
    ----------
    generated : list[str]
        Generated feedback texts.
    reference_answers : list[str]
        Reference answers per record.
    nli_pipeline : Any | None
        NLI pipeline. If ``None``, returns ``NaN`` values.

    Returns
    -------
    dict
        ``{"mean": ..., "per_record": [...]}``
    """
    if not generated or nli_pipeline is None:
        return {"mean": float("nan"), "per_record": []}

    per_record = []
    for fb, ref in zip(generated, reference_answers):
        per_record.append(
            factual_consistency_single(fb, ref, nli_pipeline)
        )

    return {
        "mean": float(np.mean(per_record)),
        "per_record": per_record,
    }


# ======================================================================
# Hallucination rate metric (Task 26.4)
# ======================================================================

def has_hallucination(
    feedback: str,
    reference_answer: str,
    nli_pipeline: Any,
) -> bool:
    """Check if feedback contains at least one contradicting claim.

    Parameters
    ----------
    feedback : str
        Generated feedback text.
    reference_answer : str
        The reference answer to check against.
    nli_pipeline : Any
        NLI pipeline.

    Returns
    -------
    bool
        ``True`` if at least one sentence contradicts the reference.
    """
    sentences = _split_sentences(feedback)
    if not sentences:
        return False

    for sentence in sentences:
        result = nli_pipeline(
            {"text": reference_answer, "text_pair": sentence},
            top_k=1,
        )
        if result and isinstance(result, list):
            top = result[0] if isinstance(result[0], dict) else result[0][0]
            label = top["label"]
        else:
            label = "neutral"

        if label in _CONTRADICTION_LABELS:
            return True

    return False


def compute_hallucination_rate(
    generated: list[str],
    reference_answers: list[str],
    *,
    nli_pipeline: Any | None = None,
) -> dict[str, float]:
    """Compute hallucination rate across generated feedback.

    Hallucination rate = fraction of records containing at least one
    claim that contradicts the reference answer.

    Parameters
    ----------
    generated : list[str]
        Generated feedback texts.
    reference_answers : list[str]
        Reference answers per record.
    nli_pipeline : Any | None
        NLI pipeline. If ``None``, returns ``NaN``.

    Returns
    -------
    dict
        ``{"rate": ..., "per_record": [...]}``
    """
    if not generated or nli_pipeline is None:
        return {"rate": float("nan"), "per_record": []}

    per_record = []
    for fb, ref in zip(generated, reference_answers):
        per_record.append(has_hallucination(fb, ref, nli_pipeline))

    rate = sum(per_record) / len(per_record)
    return {
        "rate": float(rate),
        "per_record": per_record,
    }


# ======================================================================
# Human evaluation template (Task 26.5)
# ======================================================================

_RUBRIC_DIMENSIONS = [
    "accuracy",
    "specificity",
    "actionability",
    "tone",
    "pedagogical_value",
]

_RUBRIC_DESCRIPTIONS = {
    "accuracy": (
        "How factually correct is the feedback with respect to the "
        "reference answer and key concepts?"
    ),
    "specificity": (
        "How specific is the feedback in identifying what the student "
        "got right and wrong?"
    ),
    "actionability": (
        "How actionable is the feedback — does it tell the student "
        "what to do next?"
    ),
    "tone": (
        "Is the tone encouraging, respectful, and pedagogically "
        "appropriate?"
    ),
    "pedagogical_value": (
        "Overall, how useful is this feedback for helping the student "
        "learn?"
    ),
}


def generate_human_eval_template(
    records: list[dict[str, Any]],
    n_samples: int = 100,
    stratify_by: str = "predicted_label",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a human evaluation template with a 5-point rubric.

    Each record in the output contains the question, reference answer,
    student answer, generated feedback, and a rubric section with
    5 dimensions scored 1–5.

    Parameters
    ----------
    records : list[dict]
        Records to sample from. Each dict should have at least:
        ``question``, ``reference_answer``, ``student_answer``,
        ``generated_feedback``, and the ``stratify_by`` field.
    n_samples : int
        Number of samples to include.
    stratify_by : str
        Field to stratify the sample by.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Human evaluation template records.
    """
    if not records:
        return []

    rng = np.random.default_rng(seed)

    # Group by stratification field.
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        key = str(rec.get(stratify_by, "unknown"))
        groups.setdefault(key, []).append(i)

    # Allocate samples proportionally.
    total = len(records)
    selected_indices: list[int] = []

    for key, indices in sorted(groups.items()):
        proportion = len(indices) / total
        n_from_group = max(1, round(proportion * n_samples))
        n_from_group = min(n_from_group, len(indices))
        chosen = rng.choice(
            indices, size=n_from_group, replace=False,
        ).tolist()
        selected_indices.extend(chosen)

    # Trim to n_samples if over-allocated.
    if len(selected_indices) > n_samples:
        selected_indices = rng.choice(
            selected_indices, size=n_samples, replace=False,
        ).tolist()

    # Build template.
    template: list[dict[str, Any]] = []
    for idx in sorted(selected_indices):
        rec = records[idx]
        entry: dict[str, Any] = {
            "sample_index": idx,
            "question": rec.get("question", ""),
            "reference_answer": rec.get("reference_answer", ""),
            "student_answer": rec.get("student_answer", ""),
            "generated_feedback": rec.get("generated_feedback", ""),
            "predicted_label": rec.get("predicted_label", ""),
            "rubric": {},
        }
        for dim in _RUBRIC_DIMENSIONS:
            entry["rubric"][dim] = {
                "description": _RUBRIC_DESCRIPTIONS[dim],
                "score": None,  # To be filled by human evaluator.
                "scale": "1-5 (1=very poor, 5=excellent)",
            }
        template.append(entry)

    return template


def export_human_eval_template_json(
    template: list[dict[str, Any]],
) -> str:
    """Serialize a human evaluation template to JSON string."""
    return json.dumps(template, indent=2, ensure_ascii=False)
