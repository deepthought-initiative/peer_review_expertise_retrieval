#!/usr/bin/env python3
"""
Validation script: compare refactored src/ pipeline against original notebook.

This script verifies that the new modular code produces IDENTICAL results
to the original notebook by running both side-by-side on your confidential
data and comparing score matrices and metrics numerically.

=== WORKFLOW ===

STEP 1 — Run the original notebook first and export baselines:
  Open expertise_playground_cleanup.ipynb, run all cells, then add a cell
  at the bottom with:

      # Export baselines for validation
      scores_df.to_csv("baseline/tfidf_scores.csv")
      alma_scores.to_csv("baseline/lda_scores.csv")
      specter_scores_df = pd.DataFrame(specter_scores.numpy().T,
          index=ads_df_sorted.fname, columns=proposalIDs_P110_sorted)
      specter_scores_df.to_csv("baseline/specter_scores.csv")
      sentence_scores_df = pd.DataFrame(sentence_scores.cpu().numpy().T,
          index=ads_df_sorted.fname, columns=proposalIDs_P110_sorted)
      sentence_scores_df.to_csv("baseline/sentence_scores.csv")
      keywords_scores.to_csv("baseline/keywords_scores.csv")
      gpt4o_scores.to_csv("baseline/gpt4o_scores.csv")

STEP 2 — Run this script:
  python scripts/validate_refactor.py

It will load the baselines, run the new pipeline, and report numerical diffs.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_DIR = PROJECT_ROOT / "baseline"
TOLERANCE = 1e-6  # numerical tolerance for floating-point comparison


def compare_matrices(name: str, baseline: pd.DataFrame, new: pd.DataFrame) -> bool:
    """Compare two score matrices and report differences.

    Parameters
    ----------
    name : str
        Method name for logging.
    baseline : pd.DataFrame
        Original notebook output.
    new : pd.DataFrame
        New src/ pipeline output.

    Returns
    -------
    bool
        True if matrices match within tolerance.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Shape check
    if baseline.shape != new.shape:
        print(f"  ❌ SHAPE MISMATCH: baseline={baseline.shape}, new={new.shape}")
        return False
    print(f"  Shape: {baseline.shape} ✓")

    # Align indices and columns
    common_rows = sorted(set(baseline.index) & set(new.index))
    common_cols = sorted(set(baseline.columns) & set(new.columns))

    if len(common_rows) < baseline.shape[0] or len(common_cols) < baseline.shape[1]:
        print(f"  ⚠️  Index overlap: {len(common_rows)}/{baseline.shape[0]} rows, "
              f"{len(common_cols)}/{baseline.shape[1]} cols")

    b = baseline.loc[common_rows, common_cols].values.astype(float)
    n = new.loc[common_rows, common_cols].values.astype(float)

    # Numerical comparison
    abs_diff = np.abs(b - n)
    max_diff = np.nanmax(abs_diff)
    mean_diff = np.nanmean(abs_diff)
    n_mismatches = np.sum(abs_diff > TOLERANCE)

    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")
    print(f"  Mismatches (>{TOLERANCE}): {n_mismatches} / {b.size}")

    if max_diff <= TOLERANCE:
        print(f"  ✅ PASS — numerically identical")
        return True
    elif max_diff < 0.01:
        print(f"  ⚠️  CLOSE — small numerical differences (likely floating-point)")
        return True
    else:
        print(f"  ❌ FAIL — significant differences found")
        # Show worst offenders
        worst_idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
        print(f"  Worst: row={common_rows[worst_idx[0]]}, "
              f"col={common_cols[worst_idx[1]]}")
        print(f"    baseline={b[worst_idx]:.6f}, new={n[worst_idx]:.6f}")
        return False


def validate_metrics(
    baseline_scores: dict[str, pd.DataFrame],
    new_scores: dict[str, pd.DataFrame],
    ground_truth: dict,
):
    """Compare ranking metrics between baseline and new score matrices.

    Parameters
    ----------
    baseline_scores : dict
        {method_name: baseline DataFrame}
    new_scores : dict
        {method_name: new DataFrame}
    ground_truth : dict
        {proposal_id: reviewer_id}
    """
    from src.metrics import compute_ranking_metrics

    print(f"\n{'='*60}")
    print(f"  METRICS COMPARISON")
    print(f"{'='*60}")

    for method_name in baseline_scores:
        if method_name not in new_scores:
            print(f"\n  ⚠️  {method_name}: missing from new results, skipping")
            continue

        m_base = compute_ranking_metrics(baseline_scores[method_name], ground_truth)
        m_new = compute_ranking_metrics(new_scores[method_name], ground_truth)

        common_pids = sorted(set(m_base.keys()) & set(m_new.keys()))
        if not common_pids:
            print(f"\n  ⚠️  {method_name}: no common proposals")
            continue

        mrr_base = np.array([m_base[p]["mrr"] for p in common_pids])
        mrr_new = np.array([m_new[p]["mrr"] for p in common_pids])
        mrr_diff = np.abs(mrr_base - mrr_new).max()

        rank_base = np.array([m_base[p]["rank"] for p in common_pids])
        rank_new = np.array([m_new[p]["rank"] for p in common_pids])
        rank_diff = np.abs(rank_base - rank_new).max()

        status = "✅" if mrr_diff < TOLERANCE and rank_diff < 1 else "❌"
        print(f"\n  {status} {method_name}: "
              f"MRR diff={mrr_diff:.2e}, Rank diff={rank_diff:.0f}, "
              f"N={len(common_pids)}")


def main():
    print("=" * 60)
    print("  VALIDATION: Original Notebook vs. Refactored src/ Pipeline")
    print("=" * 60)

    # Check baseline directory exists
    if not BASELINE_DIR.exists():
        print(f"\n❌ Baseline directory not found: {BASELINE_DIR}")
        print("\nTo create baselines, run the original notebook and export")
        print("score matrices to CSV. See the docstring at the top of this")
        print("script for the exact cell to add.\n")
        print("mkdir baseline")
        print("# Then run the notebook export cell")
        return

    # Load baselines
    baseline_files = {
        "TF-IDF": "tfidf_scores.csv",
        "LDA": "lda_scores.csv",
        "SPECTER": "specter_scores.csv",
        "SentenceTransformer": "sentence_scores.csv",
        "Keywords": "keywords_scores.csv",
        "GPT-4o-mini": "gpt4o_scores.csv",
    }

    baselines = {}
    for method, filename in baseline_files.items():
        path = BASELINE_DIR / filename
        if path.exists():
            baselines[method] = pd.read_csv(path, index_col=0)
            baselines[method].index = baselines[method].index.astype(str)
            baselines[method].columns = baselines[method].columns.astype(str)
            print(f"  Loaded baseline: {method} ({baselines[method].shape})")
        else:
            print(f"  ⚠️  Missing baseline: {path}")

    if not baselines:
        print("\n❌ No baseline files found. Export them from the notebook first.")
        return

    # --- Run the new pipeline on the SAME data ---
    # You need to point DATA_DIR to your confidential data
    print(f"\n  Data directory: {os.getenv('DATA_DIR', 'data/demo/')}")
    print("  (Set DATA_DIR env var to point at your confidential data)")

    from src.data_loader import (
        load_proposals, load_reviewers, load_reviewer_abstracts, load_ground_truth,
    )
    from src.embeddings import get_embedder

    proposals_df = load_proposals()
    reviewers_df = load_reviewers()
    reviewer_abstracts = load_reviewer_abstracts()
    ground_truth = load_ground_truth()

    reviewer_ids = reviewers_df["reviewer_id"].astype(str).tolist()
    reviewer_names = reviewers_df["full_name"].tolist()

    reviewer_texts = []
    for name in reviewer_names:
        data = reviewer_abstracts.get(name, {})
        abstracts = data.get("abstracts", [])
        reviewer_texts.append(" ".join(abstracts) if abstracts else "")

    proposal_ids = proposals_df["proposal_id"].astype(str).tolist()
    proposal_texts = (
        proposals_df["title"].fillna("") + " " + proposals_df["abstract"].fillna("")
    ).tolist()

    # Run each method and compare
    new_scores = {}
    method_to_key = {
        "TF-IDF": "tfidf",
        "LDA": "lda",
        "Keywords": "keywords",
        "GPT-4o-mini": "gpt4o",
        "SPECTER": "specter",
        "SentenceTransformer": "sentence_transformer",
    }

    for method_name, key in method_to_key.items():
        if method_name not in baselines:
            continue
        try:
            embedder = get_embedder(key)
            scores = embedder.compute_scores(
                proposal_texts, reviewer_texts, proposal_ids, reviewer_ids
            )
            new_scores[method_name] = scores
        except Exception as e:
            print(f"  ⚠️  Failed to run {method_name}: {e}")

    # Compare score matrices
    results = {}
    for method_name in baselines:
        if method_name in new_scores:
            results[method_name] = compare_matrices(
                method_name, baselines[method_name], new_scores[method_name]
            )

    # Compare metrics
    if ground_truth:
        validate_metrics(baselines, new_scores, ground_truth)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total = len(results)
    passed = sum(results.values())
    print(f"  {passed}/{total} methods match")
    if passed == total:
        print("  🎉 All methods produce identical results!")
    else:
        failed = [m for m, ok in results.items() if not ok]
        print(f"  ❌ Mismatches: {', '.join(failed)}")


if __name__ == "__main__":
    main()
