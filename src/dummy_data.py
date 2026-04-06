"""
Generate demo data using the ADS holdout-paper strategy.

Strategy
--------
1. **Stage 1 — Proposals & Authors**: Query ADS for the N most recent
   refereed astronomy papers.  Each paper's abstract becomes a "proposal"
   and its first-author name enters the reviewer pool (no name
   disambiguation — names are used exactly as returned by ADS, which
   mirrors how the ESO data is treated).

2. **Stage 2 — Reviewer Corpora**: For every unique first author, query
   ADS for their 25 most recent refereed papers from the last 5 years.
   The held-out proposal paper is excluded from this corpus.

3. **Ground Truth**: Each proposal maps to its first author.

Both stages are independently cached to JSON so that re-runs do not
repeat API calls that already succeeded.

Usage::

    # Full generation (requires ADS_DEV_KEY environment variable)
    python -m src.dummy_data

    # Custom size
    python -m src.dummy_data --n-recent 300

    # Explicit seed and output directory
    python -m src.dummy_data --seed 42 --output data/my_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADS query helpers
# ---------------------------------------------------------------------------
def _query_pool_papers(
    n: int = 300,
    years: int = 5,
    year: int | None = None,
    mode: str = "recent",
    cache_path: Path | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Fetch pool of papers (Recent, Year-based, or Proposals) from ADS.

    Parameters
    ----------
    n : int
        Target number of papers.
    years : int
        Look-back window (only used for 'recent' mode).
    year : int, optional
        Specific year to query (triggers 'year' mode).
    mode : str
        One of "recent", "proposals", or "year".
    cache_path : Path, optional
    seed : int, optional

    Returns
    -------
    list[dict]
        List of {first_author, title, abstract, bibcode}.
    """
    # Check cache first
    if cache_path and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= n:
            logger.info("Stage 1 cache hit: %d papers", len(cached))
            return cached[:n]

    import ads
    import datetime

    ads.config.token = config.ADS_TOKEN
    if not ads.config.token:
        raise EnvironmentError("Set ADS_DEV_KEY environment variable.")

    # 1. Build Query
    if mode == "proposals":
        q_str = 'bibstem:(hst..prop OR jwst.prop) property:not_refereed'
        sort_order = "score desc"
        kwargs = {"q": q_str, "sort": sort_order}
    elif year:
        q_str = f'year:{year} database:"astronomy"'
        sort_order = "score desc"
        kwargs = {"q": q_str, "sort": sort_order, "property": "refereed"}
    else:
        this_year = datetime.date.today().year
        q_str = f'year:{this_year - years}-{this_year} database:"astronomy"'
        sort_order = "date desc"
        kwargs = {"q": q_str, "sort": sort_order, "property": "refereed"}

    kwargs.update({
        "fl": ["first_author", "author", "title", "abstract", "bibcode"],
        "rows": max(n * 20, 2000)
    })

    logger.info("Stage 1 Query: %s", q_str)
    query = ads.SearchQuery(**kwargs)
    
    # 2. Process Results
    raw_results = []
    for paper in query:
        # Standardized PI extraction
        author_name = paper.first_author
        if not author_name and hasattr(paper, "author") and paper.author:
            author_name = paper.author[0]
            
        if paper.abstract and author_name:
            raw_results.append({
                "first_author": author_name,
                "title": (paper.title[0] if isinstance(paper.title, list) and paper.title else (paper.title or "Untitled")),
                "abstract": paper.abstract,
                "bibcode": paper.bibcode,
            })
    
    # 3. Randomize if requested
    if mode in ["proposals", "year"] or year is not None:
        import random
        rng = random.Random(seed if seed is not None else config.RANDOM_SEED)
        rng.shuffle(raw_results)
        
    papers = raw_results[:n]
    logger.info("Stage 1 Complete: %d papers (mode=%s)", len(papers), mode)

    # Persist cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

    return papers


def _fetch_author_papers(
    author_name: str,
    max_papers: int = 25,
    years: int = 5,
    exclude_bibcode: str | None = None,
    until_year: int | None = None,
) -> list[dict]:
    """Fetch recent papers for a specific author from ADS.

    Parameters
    ----------
    author_name : str
        Author name exactly as returned by ADS (no disambiguation).
    max_papers : int
        Maximum papers to retrieve.
    years : int
        Look-back window in years.
    exclude_bibcode : str, optional
        Bibcode to exclude (the held-out proposal paper).

    Returns
    -------
    list[dict]
        Papers with ``title``, ``abstract``, and ``bibcode`` keys.
    """
    import ads
    import datetime

    this_year = until_year if until_year else datetime.date.today().year
    query = ads.SearchQuery(
        author=f"^{author_name}",
        q=f'year:{this_year - years}-{this_year} database:"astronomy"',
        property="refereed",
        fl=["title", "abstract", "bibcode"],
        sort="date desc",
        rows=max_papers,
    )

    papers = []
    for paper in query:
        # Skip the held-out paper
        if exclude_bibcode and paper.bibcode == exclude_bibcode:
            continue
        title = (
            paper.title[0]
            if isinstance(paper.title, list)
            else (paper.title or "")
        )
        abstract = paper.abstract or ""
        if abstract.strip():
            papers.append(
                {"title": title, "abstract": abstract, "bibcode": paper.bibcode}
            )
    return papers


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def generate_demo_data(
    output_dir: Path | str | None = None,
    n_recent: int = 300,
    seed: int | None = None,
    max_papers_per_author: int = 25,
    lookback_years: int = 5,
    year: int | None = None,
    mode: str = "recent",
) -> Path:
    """Generate the demo dataset using the holdout-paper strategy.

    Parameters
    ----------
    output_dir : Path or str, optional
        Where to save the demo files.  Defaults to ``data/demo/``.
    n_recent : int
        Number of recent ADS papers to seed the author/proposal list from.
    seed : int, optional
        Random seed.  Defaults to ``config.RANDOM_SEED``.
    max_papers_per_author : int
        Max papers to fetch per author in stage 2.
    lookback_years : int
        How many years to look back for author papers.

    Returns
    -------
    Path
        Directory containing the generated files.
    """
    output_dir = Path(output_dir or config.DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = seed or config.RANDOM_SEED
    rng = np.random.default_rng(seed)

    # Cache paths
    stage1_cache = output_dir / "_cache_stage1_recent_papers.json"
    stage2_cache = output_dir / "_cache_stage2_author_papers.json"

    # -----------------------------------------------------------------------
    # Stage 1: Get recent papers → proposals + author names
    # -----------------------------------------------------------------------
    # Determine which papers to fetch
    # This automatically handles recent, proposals, or specific years
    recent_papers = _query_pool_papers(
        n=n_recent, 
        years=lookback_years, 
        year=year,
        mode=mode,
        cache_path=stage1_cache,
        seed=seed
    )

    # Save run metadata
    metadata = {
        "mode": mode,
        "n_target": n_recent,
        "seed": seed,
        "year": year,
        "lookback_years": lookback_years,
        "timestamp": str(pd.Timestamp.now()),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Build proposal list and deduplicate authors
    # No name disambiguation — use names exactly as returned by ADS.
    seen_authors: set[str] = set()
    proposals = []
    author_to_bibcode: dict[str, str] = {}  # track held-out bibcode
    unique_authors: list[str] = []

    for paper in recent_papers:
        name = paper["first_author"]
        if name in seen_authors:
            continue  # skip duplicate authors, keep first occurrence
        seen_authors.add(name)
        unique_authors.append(name)
        author_to_bibcode[name] = paper["bibcode"]

        proposals.append(
            {
                "first_author": name,
                "title": paper["title"],
                "abstract": paper["abstract"],
                "bibcode": paper["bibcode"],
            }
        )

    logger.info(
        "Stage 1 complete: %d unique first authors from %d papers",
        len(unique_authors),
        len(recent_papers),
    )

    # -----------------------------------------------------------------------
    # Stage 2: Fetch publication history for each author
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Fetching publication histories for %d authors...", len(unique_authors))
    logger.info("=" * 60)

    # Load existing stage 2 cache
    if stage2_cache.exists():
        with open(stage2_cache, "r", encoding="utf-8") as f:
            author_papers_cache: dict[str, list[dict]] = json.load(f)
        logger.info("Stage 2 cache loaded: %d authors cached", len(author_papers_cache))
    else:
        author_papers_cache = {}

    to_fetch = [a for a in unique_authors if a not in author_papers_cache]
    logger.info(
        "Stage 2: %d authors to fetch (%d already cached)",
        len(to_fetch),
        len(unique_authors) - len(to_fetch),
    )

    for i, author in enumerate(to_fetch, 1):
        logger.info("  [%d/%d] %s", i, len(to_fetch), author)
        try:
            papers = _fetch_author_papers(
                author,
                max_papers=max_papers_per_author,
                years=lookback_years,
                exclude_bibcode=author_to_bibcode.get(author),
                until_year=year,
            )
            author_papers_cache[author] = papers
        except Exception as e:
            logger.error("  Error fetching %s: %s", author, e)
            author_papers_cache[author] = []

        # Incremental save every 25 authors
        if i % 25 == 0:
            with open(stage2_cache, "w", encoding="utf-8") as f:
                json.dump(author_papers_cache, f, indent=2, ensure_ascii=False)
            logger.info("  (incremental cache save at %d authors)", i)

    # Final save
    with open(stage2_cache, "w", encoding="utf-8") as f:
        json.dump(author_papers_cache, f, indent=2, ensure_ascii=False)
    logger.info("Stage 2 cache saved to %s", stage2_cache)

    # -----------------------------------------------------------------------
    # Stage 3: Build output files
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Building output files...")
    logger.info("=" * 60)

    proposal_rows = []
    reviewer_rows = []
    reviewer_abstracts: dict[str, dict] = {}
    ground_truth_rows = []
    skipped = 0

    for idx, prop in enumerate(proposals):
        author = prop["first_author"]
        papers = author_papers_cache.get(author, [])

        # Need at least 1 paper in the corpus (the proposal is already held out)
        if not papers:
            logger.warning("  Skipping %s — no papers in corpus", author)
            skipped += 1
            continue

        reviewer_id = str(1000 + idx)
        proposal_id = str(5000 + idx)

        # Proposal
        proposal_rows.append(
            {
                "proposal_id": proposal_id,
                "title": prop["title"],
                "abstract": prop["abstract"],
            }
        )

        # Reviewer
        parts = author.split(", ", 1)
        last_name = parts[0] if parts else author
        first_name = parts[1] if len(parts) > 1 else ""
        reviewer_rows.append(
            {
                "reviewer_id": reviewer_id,
                "first_name": first_name,
                "last_name": last_name,
            }
        )

        # Reviewer abstracts (author's other papers)
        reviewer_abstracts[author] = {
            "abstracts": [p["abstract"] for p in papers],
            "titles": [p["title"] for p in papers],
            "count": len(papers),
        }

        # Ground truth: this proposal's true expert is this author
        ground_truth_rows.append(
            {
                "proposal_id": proposal_id,
                "reviewer_id": reviewer_id,
            }
        )

    logger.info(
        "Built %d proposals / %d reviewers (skipped %d with no corpus)",
        len(proposal_rows),
        len(reviewer_rows),
        skipped,
    )

    # Save all files
    pd.DataFrame(proposal_rows).to_csv(
        output_dir / "proposals.csv", index=False
    )
    pd.DataFrame(reviewer_rows).to_csv(
        output_dir / "reviewers.csv", index=False
    )
    pd.DataFrame(ground_truth_rows).to_csv(
        output_dir / "ground_truth.csv", index=False
    )
    with open(
        output_dir / "reviewer_abstracts.json", "w", encoding="utf-8"
    ) as f:
        json.dump(reviewer_abstracts, f, indent=2, ensure_ascii=False)

    logger.info("✅ Demo data generation complete.")
    logger.info("   Proposals:  %d", len(proposal_rows))
    logger.info("   Reviewers:  %d", len(reviewer_rows))
    logger.info("   Output dir: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generate demo data from ADS using holdout-paper strategy."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/demo/)",
    )
    parser.add_argument(
        "--n-recent",
        type=int,
        default=300,
        help="Number of recent ADS papers to query (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=25,
        help="Max papers per author in stage 2 (default: 25)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Look-back window in years (default: 5)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Specify a year for target papers (e.g. 2023). Shuffles results.",
    )
    args = parser.parse_args()

    generate_demo_data(
        output_dir=args.output,
        n_recent=args.n_recent,
        seed=args.seed,
        max_papers_per_author=args.max_papers,
        lookback_years=args.years,
        year=args.year,
    )
