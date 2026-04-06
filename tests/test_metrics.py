"""
Unit tests for ``src.metrics``.

Tests use small, hand-crafted inputs with known correct answers to verify
the MRR, Rank, Hit@K, Z-Score, and NDCG calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import compute_ranking_metrics, compute_ndcg_per_proposal, summarise_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_scores() -> pd.DataFrame:
    """3 reviewers × 2 proposals, with clear ordering.

    Proposal P1: R1 scores highest (0.9), R2 mid (0.5), R3 low (0.1)
    Proposal P2: R3 scores highest (0.8), R1 mid (0.4), R2 low (0.2)
    """
    data = {
        "P1": [0.9, 0.5, 0.1],
        "P2": [0.4, 0.2, 0.8],
    }
    return pd.DataFrame(data, index=["R1", "R2", "R3"])


@pytest.fixture
def simple_ground_truth() -> dict:
    """P1 → R1 (best match), P2 → R3 (best match)."""
    return {"P1": "R1", "P2": "R3"}


# ---------------------------------------------------------------------------
# Tests: compute_ranking_metrics
# ---------------------------------------------------------------------------
class TestComputeRankingMetrics:

    def test_perfect_ranking(self, simple_scores, simple_ground_truth):
        """When the true reviewer is the top scorer, rank=1 and MRR=1.0."""
        metrics = compute_ranking_metrics(
            simple_scores, simple_ground_truth, k_values=[1, 25], min_scores=1
        )

        # P1: R1 has the highest score → rank 1
        assert metrics["P1"]["rank"] == 1
        assert metrics["P1"]["mrr"] == pytest.approx(1.0)
        assert metrics["P1"]["hit@1"] == 1
        assert metrics["P1"]["hit@25"] == 1

        # P2: R3 has the highest score → rank 1
        assert metrics["P2"]["rank"] == 1
        assert metrics["P2"]["mrr"] == pytest.approx(1.0)

    def test_imperfect_ranking(self, simple_scores):
        """When the true reviewer is NOT the top scorer."""
        # P1's true reviewer is R2 (score 0.5, rank 2)
        gt = {"P1": "R2"}
        metrics = compute_ranking_metrics(
            simple_scores, gt, k_values=[1, 25], min_scores=1
        )
        assert metrics["P1"]["rank"] == 2
        assert metrics["P1"]["mrr"] == pytest.approx(0.5)
        assert metrics["P1"]["hit@1"] == 0
        assert metrics["P1"]["hit@25"] == 1

    def test_z_score_positive(self, simple_scores, simple_ground_truth):
        """True reviewers with above-mean scores should have positive Z."""
        metrics = compute_ranking_metrics(
            simple_scores, simple_ground_truth, k_values=[25], min_scores=1
        )
        # R1's score on P1 is 0.9, well above the mean of (0.9+0.5+0.1)/3=0.5
        assert metrics["P1"]["z"] > 0

    def test_missing_proposal_skipped(self, simple_scores):
        """Proposals not in the score matrix should be silently skipped."""
        gt = {"P_MISSING": "R1"}
        metrics = compute_ranking_metrics(
            simple_scores, gt, k_values=[25], min_scores=1
        )
        assert len(metrics) == 0

    def test_min_scores_filter(self, simple_scores, simple_ground_truth):
        """Proposals with fewer reviewers than min_scores are skipped."""
        metrics = compute_ranking_metrics(
            simple_scores, simple_ground_truth, k_values=[25], min_scores=100
        )
        assert len(metrics) == 0


# ---------------------------------------------------------------------------
# Tests: NDCG
# ---------------------------------------------------------------------------
class TestNDCG:

    def test_perfect_ndcg(self):
        """Perfect ranking gives NDCG = 1.0."""
        df = pd.DataFrame({
            "proposal_id": ["P1", "P1", "P1"],
            "score": [0.9, 0.5, 0.1],
            "expertise_label": ["Expert", "Intermediate", "Non-Expert"],
        })
        result = compute_ndcg_per_proposal(df, score_column="score")
        assert result.iloc[0]["ndcg"] == pytest.approx(1.0)

    def test_inverted_ranking_lower_ndcg(self):
        """Inverted ranking gives NDCG < 1.0."""
        df = pd.DataFrame({
            "proposal_id": ["P1", "P1", "P1"],
            "score": [0.1, 0.5, 0.9],  # Non-Expert ranked highest
            "expertise_label": ["Expert", "Intermediate", "Non-Expert"],
        })
        result = compute_ndcg_per_proposal(df, score_column="score")
        assert result.iloc[0]["ndcg"] < 1.0


# ---------------------------------------------------------------------------
# Tests: summarise_metrics
# ---------------------------------------------------------------------------
class TestSummariseMetrics:

    def test_output_shape(self, simple_scores, simple_ground_truth):
        """Summary table should have one row per method."""
        m = compute_ranking_metrics(
            simple_scores, simple_ground_truth, k_values=[25], min_scores=1
        )
        all_metrics = {"TF-IDF": m, "LDA": m}
        summary = summarise_metrics(all_metrics)
        assert len(summary) == 2
        assert "Method" in summary.columns
        assert "Mean MRR" in summary.columns
