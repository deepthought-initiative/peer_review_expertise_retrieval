"""
Reporting utilities for the expertise evaluation pipeline.

Produces summary tables (pandas DataFrames) and optionally LaTeX-formatted
output suitable for direct inclusion in a journal manuscript.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src import config


# =========================================================================
# Bootstrap confidence interval helper
# =========================================================================
def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observations.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95%).
    seed : int, optional
        Random seed.  Defaults to ``config.RANDOM_SEED``.

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of the CI.
    """
    rng = np.random.default_rng(seed or config.RANDOM_SEED)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


# =========================================================================
# Report: single method
# =========================================================================
def report_method(
    metrics_dict: dict[str, dict],
    method_name: str,
) -> dict:
    """Aggregate per-proposal metrics into a summary row.

    Parameters
    ----------
    metrics_dict : dict
        ``{proposal_id: {"mrr": ..., "rank": ..., "z": ..., "hit@25": ...}}``.
    method_name : str
        Human-readable method name (for the resulting row).

    Returns
    -------
    dict
        Summary statistics including mean, median, and 95% CI.
    """
    ids = sorted(metrics_dict.keys())
    mrrs = np.array([metrics_dict[pid]["mrr"] for pid in ids])
    ranks = np.array([metrics_dict[pid]["rank"] for pid in ids])
    zs = np.array([metrics_dict[pid]["z"] for pid in ids])
    hits25 = np.array([metrics_dict[pid].get("hit@25", 0) for pid in ids])

    mrr_ci = _bootstrap_ci(mrrs)
    z_ci = _bootstrap_ci(zs)

    return {
        "Method": method_name,
        "MRR": f"{mrrs.mean():.3f}",
        "MRR 95% CI": f"[{mrr_ci[0]:.3f}, {mrr_ci[1]:.3f}]",
        "Median Rank": f"{np.median(ranks):.0f}",
        "Mean Rank": f"{ranks.mean():.1f}",
        "Hit@25": f"{hits25.mean():.3f}",
        "Z-Score": f"{zs.mean():.3f}",
        "Z 95% CI": f"[{z_ci[0]:.3f}, {z_ci[1]:.3f}]",
        "N": len(ids),
    }


# =========================================================================
# Report: all methods in one table
# =========================================================================
def report_all_methods(
    all_metrics: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """Produce a comparative summary table across all methods.

    Parameters
    ----------
    all_metrics : dict
        ``{method_name: {proposal_id: {metric_key: value, ...}, ...}}``.

    Returns
    -------
    pd.DataFrame
        One row per method with summary statistics.
    """
    rows = [
        report_method(m_dict, name)
        for name, m_dict in all_metrics.items()
    ]
    return pd.DataFrame(rows)


# =========================================================================
# LaTeX formatting
# =========================================================================
def to_latex(
    summary_df: pd.DataFrame,
    caption: str = "Summary of ranking metrics across expertise methods.",
    label: str = "tab:metrics_summary",
) -> str:
    """Convert a summary DataFrame to a LaTeX table string.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`report_all_methods`.
    caption : str
        Table caption.
    label : str
        LaTeX label for cross-referencing.

    Returns
    -------
    str
        LaTeX source for the table.
    """
    latex = summary_df.to_latex(
        index=False,
        caption=caption,
        label=label,
        column_format="l" + "r" * (len(summary_df.columns) - 1),
        escape=True,
    )
    return latex
