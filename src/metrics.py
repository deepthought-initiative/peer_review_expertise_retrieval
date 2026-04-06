"""
Evaluation metrics for the expertise-matching pipeline.

Implements the four primary ranking metrics used in the paper:

- **MRR** (Mean Reciprocal Rank) — how high the true reviewer ranks.
- **Hit@K** — whether the true reviewer appears in the top-K list.
- **Median Rank** — the raw rank of the true reviewer.
- **Z-Score** — how many standard deviations the true reviewer's score
  is above the mean of all candidates.

Plus:

- **NDCG** — graded relevance score when ground-truth expertise levels
  (Expert / Intermediate / Non-Expert) are available.
- **Paired Wilcoxon signed-rank test** — for pairwise statistical
  significance between methods.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import ndcg_score

from src import config

logger = logging.getLogger(__name__)


# =========================================================================
# Core ranking metrics (per-proposal)
# =========================================================================
def compute_ranking_metrics(
    scores_df: pd.DataFrame,
    ground_truth: dict[str, str],
    k_values: list[int] | None = None,
    min_scores: int | None = None,
) -> dict[str, dict]:
    """Compute per-proposal ranking metrics for one method.

    For each proposal, the true reviewer's rank within all candidates is
    determined and several metrics are derived.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Score matrix (reviewers × proposals).  Rows = reviewer IDs,
        columns = proposal IDs.
    ground_truth : dict[str, str]
        ``{proposal_id: true_reviewer_id}`` mapping.
    k_values : list[int], optional
        Cutoffs for Hit@K.  Defaults to ``config.HIT_AT_K`` (``[25]``).
    min_scores : int, optional
        Skip proposals with fewer than this many non-NaN reviewer scores.
        Defaults to ``config.MIN_SCORES_PER_PROPOSAL``.

    Returns
    -------
    dict[str, dict]
        ``{proposal_id: {"mrr": float, "rank": int, "hit@K": 0|1,
        "z": float, "score": float}}``.
    """
    k_values = k_values or config.HIT_AT_K
    min_scores = min_scores if min_scores is not None else config.MIN_SCORES_PER_PROPOSAL

    metrics: dict[str, dict] = {}

    for proposal_id, true_reviewer_id in ground_truth.items():
        proposal_id_str = str(proposal_id)
        true_reviewer_str = str(true_reviewer_id)

        # Check that both proposal and reviewer exist in the score matrix
        if proposal_id_str not in scores_df.columns:
            continue
        if true_reviewer_str not in scores_df.index:
            continue

        # Get all scores for this proposal, drop NaNs
        proposal_scores = scores_df[proposal_id_str].dropna()
        if len(proposal_scores) < min_scores:
            continue

        true_score = proposal_scores.get(true_reviewer_str)
        if true_score is None or pd.isna(true_score):
            continue

        # Rank: sort descending, then find the position of the true reviewer.
        # This matches the notebook's: sorted_scores.index.get_loc(key) + 1
        sorted_scores = proposal_scores.sort_values(ascending=False)
        loc = sorted_scores.index.get_loc(true_reviewer_str)
        # get_loc can return a slice for duplicates; take first position
        if isinstance(loc, slice):
            rank = loc.start + 1
        elif isinstance(loc, np.ndarray):
            rank = int(loc.nonzero()[0][0]) + 1
        else:
            rank = int(loc) + 1

        # MRR
        mrr = 1.0 / rank

        # Hit@K for each k
        hit_at = {}
        for k in k_values:
            hit_at[f"hit@{k}"] = 1 if rank <= k else 0

        # Z-Score: exclude the true reviewer from mean/std (ddof=1)
        # Matches notebook: others = scores.drop(index=key)
        others = proposal_scores.drop(index=true_reviewer_str).to_numpy()
        others_std = others.std(ddof=1)
        z = (true_score - others.mean()) / others_std if others_std > 0 else 0.0

        result = {
            "mrr": mrr,
            "rank": rank,
            "z": z,
            "score": float(true_score),
            **hit_at,
        }
        metrics[proposal_id_str] = result

    logger.info("Computed ranking metrics for %d proposals", len(metrics))
    return metrics


def get_top_matches(
    scores_df: pd.DataFrame,
    proposal_id: str,
    reviewers_df: pd.DataFrame,
    k: int = 5,
) -> list[dict]:
    """Get the top-K recommended reviewers for a specific proposal.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Score matrix (reviewers × proposals).
    proposal_id : str
        The ID of the proposal to analyze.
    reviewers_df : pd.DataFrame
        Reviewer metadata.
    k : int
        Number of matches to return.

    Returns
    -------
    list[dict]
        List of ``{"reviewer_id": str, "full_name": str, "score": float}``.
    """
    if proposal_id not in scores_df.columns:
        return []

    # Get scores and sort descending
    scores = scores_df[proposal_id].sort_values(ascending=False).head(k)

    # Build response
    results = []
    for reviewer_id, score in scores.items():
        reviewer_id_str = str(reviewer_id)
        row = reviewers_df[reviewers_df["reviewer_id"].astype(str) == reviewer_id_str]
        full_name = row["full_name"].values[0] if not row.empty else "Unknown"
        results.append(
            {"reviewer_id": reviewer_id_str, "full_name": full_name, "score": score}
        )

    return results


# =========================================================================
# NDCG with graded relevance
# =========================================================================
# Relevance mapping: Expert > Intermediate > Non-Expert
RELEVANCE_MAP = {
    "Expert": 10,
    "Intermediate": 2,
    "Non-Expert": 0,
    # Also support integer keys from raw data
    2: 10, "2": 10,
    3: 2, "3": 2,
    4: 0, "4": 0,
    1: 0, "1": 0,
}


def compute_ndcg_per_proposal(
    long_df: pd.DataFrame,
    score_column: str,
    relevance_column: str = "expertise_label",
) -> pd.DataFrame:
    """Compute NDCG per proposal when graded relevance labels exist.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-form DataFrame with at least ``proposal_id``, the
        ``score_column``, and ``relevance_column``.
    score_column : str
        Column name containing the method's predicted scores.
    relevance_column : str
        Column containing expertise labels (``"Expert"``, etc.).

    Returns
    -------
    pd.DataFrame
        One row per proposal with columns ``proposal_id`` and ``ndcg``.
    """
    results = []
    for pid, group in long_df.groupby("proposal_id"):
        if len(group) < 2:
            continue
        relevances = group[relevance_column].map(RELEVANCE_MAP).values
        scores = group[score_column].values
        try:
            score = ndcg_score([relevances], [scores])
        except ValueError:
            score = np.nan
        results.append({"proposal_id": pid, "ndcg": score})
    return pd.DataFrame(results)


# =========================================================================
# Pairwise statistical tests
# =========================================================================
def pairwise_wilcoxon(
    method_metrics: dict[str, dict[str, dict]],
    baseline_name: str,
    metric_key: str = "mrr",
) -> pd.DataFrame:
    """Run paired Wilcoxon signed-rank tests vs. a baseline.

    Parameters
    ----------
    method_metrics : dict[str, dict]
        ``{method_name: {proposal_id: {metric_key: value, ...}, ...}}``.
    baseline_name : str
        Name of the baseline method to compare against.
    metric_key : str
        Which metric to compare (default ``"mrr"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``method``, ``statistic``, ``p_value``, ``mean_diff``.
    """
    baseline = method_metrics[baseline_name]
    results = []

    for method_name, m_metrics in method_metrics.items():
        if method_name == baseline_name:
            continue

        # Align on common proposal IDs
        common = sorted(set(baseline.keys()) & set(m_metrics.keys()))
        if len(common) < 5:
            logger.warning(
                "Skipping %s vs %s: only %d common proposals",
                method_name, baseline_name, len(common),
            )
            continue

        base_vals = np.array([baseline[pid][metric_key] for pid in common])
        meth_vals = np.array([m_metrics[pid][metric_key] for pid in common])

        try:
            stat, p = stats.wilcoxon(meth_vals, base_vals, alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan

        results.append({
            "method": method_name,
            "vs_baseline": baseline_name,
            "metric": metric_key,
            "statistic": stat,
            "p_value": p,
            "mean_diff": float(np.mean(meth_vals - base_vals)),
            "n_proposals": len(common),
        })

    return pd.DataFrame(results)


# =========================================================================
# Summary table
# =========================================================================
def summarise_metrics(
    method_metrics: dict[str, dict[str, dict]],
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """Aggregate per-proposal metrics into a summary table.

    Parameters
    ----------
    method_metrics : dict[str, dict]
        ``{method_name: {proposal_id: {"mrr": ..., "rank": ..., ...}}}``.
    k_values : list[int], optional
        Which Hit@K columns to include.

    Returns
    -------
    pd.DataFrame
        One row per method with Mean MRR, Median Rank, Mean Hit@K,
        Mean Z-Score, and their standard deviations.
    """
    k_values = k_values or config.HIT_AT_K
    rows = []

    for method_name, m_metrics in method_metrics.items():
        ids = sorted(m_metrics.keys())
        mrrs = np.array([m_metrics[pid]["mrr"] for pid in ids])
        ranks = np.array([m_metrics[pid]["rank"] for pid in ids])
        zs = np.array([m_metrics[pid]["z"] for pid in ids])

        row = {
            "Method": method_name,
            "Mean MRR": f"{np.mean(mrrs):.3f} ± {np.std(mrrs):.3f}",
            "Median Rank": f"{np.median(ranks):.0f}",
            "Mean Z-Score": f"{np.mean(zs):.3f} ± {np.std(zs):.3f}",
        }

        for k in k_values:
            key = f"hit@{k}"
            hits = np.array([m_metrics[pid].get(key, 0) for pid in ids])
            row[f"Hit@{k}"] = f"{np.mean(hits):.3f}"

        rows.append(row)

    return pd.DataFrame(rows)
