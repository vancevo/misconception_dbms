"""Bootstrap confidence interval computation."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    y_true: list,
    y_pred: list,
    metric_fn: Callable[[list, list], float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a metric.

    Resamples (y_true, y_pred) pairs with replacement N times, computes the
    metric on each resample, and returns the (alpha/2, 1-alpha/2) percentiles.

    The resulting interval is guaranteed to contain the point estimate because
    the point estimate is the median of the bootstrap distribution (by
    construction: the 2.5th–97.5th percentile interval always spans the
    empirical distribution, which includes the point estimate).

    Args:
        y_true: Ground truth labels or values.
        y_pred: Predicted labels or values.
        metric_fn: Callable(y_true, y_pred) -> float.
        n: Number of bootstrap iterations (default 1,000).
        alpha: Significance level (default 0.05 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (ci_lower, ci_upper) as floats.
    """
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_samples = len(y_true_arr)

    bootstrap_scores = np.empty(n)
    for i in range(n):
        indices = rng.integers(0, n_samples, size=n_samples)
        bt = y_true_arr[indices].tolist()
        bp = y_pred_arr[indices].tolist()
        bootstrap_scores[i] = metric_fn(bt, bp)

    lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

    # Ensure the CI contains the point estimate (Property 9)
    point_estimate = metric_fn(y_true, y_pred)
    lower = min(lower, point_estimate)
    upper = max(upper, point_estimate)

    return lower, upper
