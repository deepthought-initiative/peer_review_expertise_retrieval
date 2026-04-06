"""
Smoke tests for ``src.embeddings``.

These tests verify that the TF-IDF and LDA embedders produce valid output
(correct shape, no NaNs) and are deterministic when given a fixed random seed.
Transformer-based embedders are not tested here to avoid slow model downloads
in CI — use ``@pytest.mark.slow`` for those if desired.
"""

import numpy as np
import pytest

from src.embeddings import TfidfEmbedder, LdaEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_corpus():
    """Small astronomy-themed corpus for smoke tests."""
    proposals = [
        "Black hole accretion disk X-ray binary system.",
        "Exoplanet atmosphere characterisation transit spectroscopy.",
        "Galaxy cluster dark matter gravitational lensing.",
    ]
    reviewers = [
        "Active galactic nuclei jet emission relativistic plasma.",
        "Planetary atmospheres chemical abundance spectroscopic analysis.",
        "Cosmological simulations large scale structure baryon acoustic.",
        "Stellar evolution nucleosynthesis white dwarf supernova progenitor.",
    ]
    proposal_ids = ["P1", "P2", "P3"]
    reviewer_ids = ["R1", "R2", "R3", "R4"]
    return proposals, reviewers, proposal_ids, reviewer_ids


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
class TestTfidfEmbedder:

    def test_output_shape(self, toy_corpus):
        """Output should be (n_reviewers × n_proposals)."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = TfidfEmbedder()
        scores = emb.compute_scores(proposals, reviewers, pids, rids)
        assert scores.shape == (len(rids), len(pids))

    def test_no_nans(self, toy_corpus):
        """No NaN values in the output."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = TfidfEmbedder()
        scores = emb.compute_scores(proposals, reviewers, pids, rids)
        assert not scores.isna().any().any()

    def test_values_in_range(self, toy_corpus):
        """Cosine similarity should be in [0, 1] for TF-IDF (non-negative)."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = TfidfEmbedder()
        scores = emb.compute_scores(proposals, reviewers, pids, rids)
        assert (scores.values >= -0.01).all()  # numerical tolerance
        assert (scores.values <= 1.01).all()

    def test_deterministic(self, toy_corpus):
        """Two runs with same seed should produce identical results."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = TfidfEmbedder()
        s1 = emb.compute_scores(proposals, reviewers, pids, rids)
        s2 = emb.compute_scores(proposals, reviewers, pids, rids)
        np.testing.assert_array_equal(s1.values, s2.values)


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------
class TestLdaEmbedder:

    def test_output_shape(self, toy_corpus):
        """Output should be (n_reviewers × n_proposals)."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = LdaEmbedder(num_topics=3, passes=2, workers=1)
        scores = emb.compute_scores(proposals, reviewers, pids, rids)
        assert scores.shape == (len(rids), len(pids))

    def test_no_nans(self, toy_corpus):
        """No NaN values in the output."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = LdaEmbedder(num_topics=3, passes=2, workers=1)
        scores = emb.compute_scores(proposals, reviewers, pids, rids)
        assert not scores.isna().any().any()

    def test_deterministic(self, toy_corpus):
        """LDA with fixed random_state should be deterministic."""
        proposals, reviewers, pids, rids = toy_corpus
        emb = LdaEmbedder(num_topics=3, passes=2, workers=1)
        s1 = emb.compute_scores(proposals, reviewers, pids, rids)
        s2 = emb.compute_scores(proposals, reviewers, pids, rids)
        np.testing.assert_array_almost_equal(s1.values, s2.values, decimal=5)
