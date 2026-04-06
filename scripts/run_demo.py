#!/usr/bin/env python3
"""
End-to-end demo of the expertise-matching pipeline on simulated ADS data.

This script demonstrates the full pipeline by:
1. Generating a simulated reviewer–proposal dataset from NASA/ADS
   (or reusing previously cached data).
2. Running the four compute-based embedding methods (TF-IDF, LDA,
   SPECTER2, SentenceTransformer) to produce similarity matrices.
3. Evaluating each method's ability to rank the true author (first
   author of the paper) as the top expert for their own proposal.
4. Reporting summary metrics: MRR, Hit@25, Median Rank, Z-Score.

Keywords and GPT-4o-mini are skipped because they require pre-computed
score matrices. See the paper for details on how these are generated.

NDCG is shown as a placeholder — it requires self-reported expertise
labels (Expert/Intermediate/Non-Expert) which are not available in the
simulated dataset. The code structure is included so users can see how
NDCG would be invoked if labels were available.

Prerequisites:
    export ADS_DEV_KEY="your-ads-api-key"

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --n-recent 50   # smaller demo
    python scripts/run_demo.py --skip-fetch     # reuse cached data
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import (
    load_ground_truth,
    load_proposals,
    load_reviewer_abstracts,
    load_reviewers,
)
from src.embeddings import (
    LdaEmbedder,
    SentenceTransformerEmbedder,
    SpecterEmbedder,
    TfidfEmbedder,
)
from src.metrics import compute_ranking_metrics, compute_ndcg_per_proposal, RELEVANCE_MAP
from src.reporting import report_all_methods

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


def get_demo_dir(mode: str, year: int | None, seed: int) -> Path:
    """Generate a descriptive directory name for the demo run."""
    base = config.DATA_DIR.parent
    if mode == "proposals":
        name = f"demo_proposals_seed_{seed}"
    elif year:
        name = f"demo_{year}_seed_{seed}"
    else:
        name = f"demo_recent_seed_{seed}"
    return base / name


def run_demo(
    data_dir: Path,
    n_recent: int = 300,
    skip_fetch: bool = False,
    methods: list[str] | None = None,
    year: int | None = None,
    mode: str = "recent",
    seed: int | None = None,
):
    """Run the full expertise-matching demo.

    Parameters
    ----------
    data_dir : Path
        Directory containing (or to contain) the demo data files.
    n_recent : int
        Number of recent papers to query if generating data.
    skip_fetch : bool
        If True, skip data generation and use existing files.
    methods : list[str], optional
        Which methods to run.  Defaults to all four compute-based methods.
    """
    # ------------------------------------------------------------------
    # Step 1: Generate or load data
    # ------------------------------------------------------------------
    proposals_path = data_dir / "proposals.csv"

    if skip_fetch and proposals_path.exists():
        print(f"\n{'='*60}")
        print("STEP 1: Using existing data (--skip-fetch)")
        print(f"{'='*60}")
        print(f"  Data directory: {data_dir}")
    else:
        print(f"\n{'='*60}")
        print(f"STEP 1: Generating simulated dataset ({mode} mode)")
        print(f"{'='*60}")

        from src.dummy_data import generate_demo_data

        generate_demo_data(
            output_dir=data_dir,
            n_recent=n_recent,
            year=year,
            mode=mode,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Step 2: Load data using the standard src/ loaders
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: Loading data")
    print(f"{'='*60}")

    proposals_df = load_proposals(data_dir / "proposals.csv")
    reviewers_df = load_reviewers(data_dir / "reviewers.csv")
    reviewer_abstracts = load_reviewer_abstracts(
        data_dir / "reviewer_abstracts.json"
    )
    ground_truth = load_ground_truth(data_dir / "ground_truth.csv")

    print(f"  Proposals:         {len(proposals_df)}")
    print(f"  Reviewers:         {len(reviewers_df)}")
    print(f"  Ground truth:      {len(ground_truth)} pairs")
    print(f"  Reviewer corpora:  {len(reviewer_abstracts)} authors")

    # ------------------------------------------------------------------
    # Step 3: Prepare text corpora
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: Preparing text corpora")
    print(f"{'='*60}")

    reviewer_ids = reviewers_df["reviewer_id"].astype(str).tolist()
    reviewer_names = reviewers_df["full_name"].tolist()

    # Build reviewer paper lists (list of lists, matching embedder API)
    reviewer_paper_lists: list[list[str]] = []
    for name in reviewer_names:
        data = reviewer_abstracts.get(name, {})
        abstracts = data.get("abstracts", [])
        reviewer_paper_lists.append(abstracts if abstracts else [])

    # Proposal texts
    proposal_ids = proposals_df["proposal_id"].astype(str).tolist()
    proposal_texts = (
        proposals_df["title"].fillna("") + " " + proposals_df["abstract"].fillna("")
    ).tolist()

    print(f"  Reviewer texts prepared: {len(reviewer_paper_lists)}")
    print(f"  Proposal texts prepared: {len(proposal_texts)}")

    # ------------------------------------------------------------------
    # Step 4: Run embedding methods
    # ------------------------------------------------------------------
    available_methods = {
        "TF-IDF": TfidfEmbedder(),
        "LDA": LdaEmbedder(),
        "SPECTER2": SpecterEmbedder(),
        "SentenceTransformer": SentenceTransformerEmbedder(),
    }

    if methods is None:
        methods = list(available_methods.keys())

    print(f"\n{'='*60}")
    print(f"STEP 4: Running {len(methods)} embedding methods")
    print(f"{'='*60}")
    print()
    print("  NOTE: Keywords and GPT-4o-mini are skipped in this demo because")
    print("  they require pre-computed score matrices. See the paper for how")
    print("  these scores are generated (keyword overlap from the observatory")
    print("  system and OpenAI structured-output API respectively).")
    print()

    score_matrices: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}

    for method_name in methods:
        if method_name not in available_methods:
            print(f"  ⚠️  Unknown method: {method_name}, skipping")
            continue

        embedder = available_methods[method_name]
        print(f"  Running: {method_name}...")

        t0 = time.time()
        scores = embedder.compute_scores(
            proposal_texts, reviewer_paper_lists, proposal_ids, reviewer_ids
        )
        elapsed = time.time() - t0

        score_matrices[method_name] = scores
        timings[method_name] = elapsed
        print(f"    Shape: {scores.shape}  |  Time: {elapsed:.1f}s")

    print(f"\n  Completed {len(score_matrices)} methods.")

    # ------------------------------------------------------------------
    # Step 5: Compute ranking metrics
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 5: Evaluating ranking metrics")
    print(f"{'='*60}")

    all_metrics: dict[str, dict[str, dict]] = {}

    for method_name, scores_df in score_matrices.items():
        m = compute_ranking_metrics(scores_df, ground_truth, min_scores=1)
        all_metrics[method_name] = m
        print(f"  {method_name}: {len(m)} proposals evaluated")

    # Summary table
    summary = report_all_methods(all_metrics)
    print(f"\n  Summary Table:")
    print(f"  {'─'*70}")
    print(summary.to_string(index=False))
    print(f"  {'─'*70}")

    # ------------------------------------------------------------------
    # Step 6: NDCG placeholder
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 6: NDCG (requires self-reported expertise labels)")
    print(f"{'='*60}")
    print()
    print("  NDCG evaluation is NOT performed in this demo because it")
    print("  requires self-reported expertise labels (Expert / Intermediate")
    print("  / Non-Expert) from each reviewer for each proposal.")
    print()
    print("  If you have expertise labels, prepare a long-form DataFrame with:")
    print("    - proposal_id, account_id, score, expertise_label")
    print("  Then call:")
    print()
    print("    from src.metrics import compute_ndcg_per_proposal, RELEVANCE_MAP")
    print("    ndcg_results = compute_ndcg_per_proposal(")
    print("        long_df, score_column='score', relevance_column='expertise_label'")
    print("    )")
    print()
    print(f"  Relevance weights: {dict((k,v) for k,v in RELEVANCE_MAP.items() if isinstance(k, str) and k[0].isupper())}")
    print()

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    output_path = config.OUTPUT_DIR / "demo_results.csv"
    summary.to_csv(output_path, index=False)
    print(f"{'='*60}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*60}")

    return summary, all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run the expertise-matching demo on simulated ADS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Data Generation Args
    data_group = parser.add_argument_group("Data Generation")
    data_group.add_argument(
        "--n-recent", type=int, default=300, help="Number of papers/proposals to sample"
    )
    data_group.add_argument(
        "--year", type=int, default=None, help="Target a specific year (triggers randomized year mode)"
    )
    data_group.add_argument(
        "--proposals", action="store_true", help="Use HST/JWST proposal records (PI -> Refereed history)"
    )
    data_group.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility"
    )
    data_group.add_argument(
        "--skip-fetch", action="store_true", help="Skip ADS API calls and use existing directory"
    )
    
    # 2. Model Execution Args
    model_group = parser.add_argument_group("Model Execution")
    model_group.add_argument(
        "--methods", nargs="+", default=None, help="Specific methods to run",
        choices=["TF-IDF", "LDA", "SPECTER2", "SentenceTransformer"]
    )
    
    args = parser.parse_args()

    # Determine mode and directory
    mode = "proposals" if args.proposals else ("year" if args.year else "recent")
    data_dir = get_demo_dir(mode, args.year, args.seed)

    run_demo(
        data_dir=data_dir,
        n_recent=args.n_recent,
        skip_fetch=args.skip_fetch,
        methods=args.methods,
        year=args.year,
        mode=mode,
        seed=args.seed,
    )
