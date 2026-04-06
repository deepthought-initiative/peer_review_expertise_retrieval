from __future__ import annotations
"""
Data loading utilities for the expertise evaluation pipeline.

Provides functions to load proposals, reviewers, ground-truth labels,
and pre-computed score matrices from the standardised directory layout
defined in ``src.config``.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------
def load_proposals(path: Path | str | None = None) -> pd.DataFrame:
    """Load the proposals dataset.

    Expected CSV columns: ``proposal_id``, ``title``, ``abstract``.

    Parameters
    ----------
    path : Path or str, optional
        Path to the proposals CSV.  Defaults to ``config.PROPOSALS_CSV``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``proposal_id``, ``title``, ``abstract``.
    """
    path = Path(path) if path else config.PROPOSALS_CSV
    logger.info("Loading proposals from %s", path)
    df = pd.read_csv(path)
    required = {"proposal_id", "title", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Proposals CSV missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Reviewers
# ---------------------------------------------------------------------------
def load_reviewers(path: Path | str | None = None) -> pd.DataFrame:
    """Load the reviewers dataset and build a ``full_name`` column.

    Expected CSV columns: ``reviewer_id``, ``first_name``, ``last_name``.

    Parameters
    ----------
    path : Path or str, optional
        Path to the reviewers CSV.  Defaults to ``config.REVIEWERS_CSV``.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``full_name`` column (``"Last, First"``).
    """
    path = Path(path) if path else config.REVIEWERS_CSV
    logger.info("Loading reviewers from %s", path)
    df = pd.read_csv(path)
    # Build canonical "Last, First" name used as the index in score matrices
    df["full_name"] = (
        df["last_name"].str.strip() + ", " + df["first_name"].str.strip()
    )
    return df


# ---------------------------------------------------------------------------
# Reviewer Abstracts (ADS cache)
# ---------------------------------------------------------------------------
def load_reviewer_abstracts(
    path: Path | str | None = None,
) -> dict[str, dict]:
    """Load the cached reviewer-abstracts JSON.

    The file maps ``full_name -> {"abstracts": [...], "titles": [...], ...}``.

    Parameters
    ----------
    path : Path or str, optional
        Defaults to ``config.REVIEWER_ABSTRACTS_JSON``.

    Returns
    -------
    dict
        ``{reviewer_name: {"abstracts": list[str], "titles": list[str], ...}}``.
    """
    path = Path(path) if path else config.REVIEWER_ABSTRACTS_JSON
    logger.info("Loading reviewer abstracts from %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------
def load_ground_truth(
    path: Path | str | None = None,
) -> dict[str, str]:
    """Build a proposal → reviewer ground-truth mapping.

    Expected CSV columns: ``proposal_id``, ``reviewer_id``.

    Parameters
    ----------
    path : Path or str, optional
        Defaults to ``config.GROUND_TRUTH_CSV``.

    Returns
    -------
    dict
        ``{proposal_id: reviewer_id}`` mapping.
    """
    path = Path(path) if path else config.GROUND_TRUTH_CSV
    logger.info("Loading ground truth from %s", path)
    df = pd.read_csv(path)
    # Each proposal maps to one true reviewer in the holdout strategy
    return dict(zip(df["proposal_id"].astype(str), df["reviewer_id"].astype(str)))


# ---------------------------------------------------------------------------
# Pre-computed Score Matrices
# ---------------------------------------------------------------------------
def load_precomputed_scores(path: Path | str) -> pd.DataFrame:
    """Load a pre-computed reviewer × proposal score matrix from CSV.

    Supports both wide-form (matrix) and long-form CSVs (account_id, 
    proposal_id, score).  Automatically pivots long-form data.

    Parameters
    ----------
    path : Path or str
        Path to the score CSV.

    Returns
    -------
    pd.DataFrame
        Score matrix with reviewers as rows, proposals as columns.
    """
    path = Path(path)
    logger.info("Loading pre-computed scores from %s", path)
    
    # Try reading without index first to check for long-form
    df = pd.read_csv(path)
    
    # Detect long-form (Keywords or GPT formats)
    cols = [c.lower() for c in df.columns]
    if "account_id" in cols and "proposal_id" in cols:
        # Keywords format - deduplicate by mean
        val_col = [c for c in df.columns if "score" in c.lower()][0]
        df = df.groupby(["account_id", "proposal_id"])[val_col].mean().reset_index()
        df = df.pivot(index="account_id", columns="proposal_id", values=val_col)
    elif "reviewer" in cols and "proposal" in cols:
        # GPT format - deduplicate by mean
        val_col = [c for c in df.columns if "score" in c.lower()][0]
        df = df.groupby(["Reviewer", "Proposal"])[val_col].mean().reset_index()
        df = df.pivot(index="Reviewer", columns="Proposal", values=val_col)
    else:
        # Assume wide-form, reload with index 0
        df = pd.read_csv(path, index_col=0)

    # Clean up indices/columns
    df.columns = df.columns.astype(str)
    df.index = df.index.astype(str)
    return df
