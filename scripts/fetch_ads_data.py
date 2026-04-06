#!/usr/bin/env python3
"""
Standalone ADS data-fetching utility.

This script fetches paper data (titles + abstracts) for a list of author
names from NASA ADS and caches the results to a JSON file.  It is NOT
part of the main pipeline — it exists as a convenience for users who want
to regenerate or extend the demo data.

Prerequisites:
    export ADS_DEV_KEY="your-ads-api-key"

Usage:
    python scripts/fetch_ads_data.py
    python scripts/fetch_ads_data.py --output output/my_abstracts.json
    python scripts/fetch_ads_data.py --names "Einstein, Albert" "Hawking, Stephen"
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_author_papers(
    author_name: str,
    token: str,
    max_papers: int = 25,
    recent_years: int = 5,
) -> dict:
    """Fetch recent papers for a single author from ADS.

    Parameters
    ----------
    author_name : str
        Author name in ``"Last, First"`` format.
    token : str
        ADS API token.
    max_papers : int
        Maximum number of papers to retrieve.
    recent_years : int
        Look-back window in years.

    Returns
    -------
    dict
        ``{"abstracts": [...], "titles": [...], "count": int}``.
    """
    import ads

    ads.config.token = token
    this_year = datetime.date.today().year

    query = ads.SearchQuery(
        author=f'"{author_name}"',
        q=f"year:{this_year - recent_years}-{this_year}",
        property="refereed",
        fl=["title", "abstract", "bibcode"],
        sort="date desc",
        rows=max_papers,
    )

    abstracts, titles = [], []
    for paper in query:
        if paper.abstract:
            abstracts.append(paper.abstract)
        if paper.title:
            t = paper.title[0] if isinstance(paper.title, list) else paper.title
            titles.append(t)

    return {"abstracts": abstracts, "titles": titles, "count": len(abstracts)}


def fetch_all(
    author_names: list[str],
    cache_path: Path | str,
    token: str,
    max_papers: int = 25,
) -> dict:
    """Fetch data for multiple authors, using a JSON cache.

    Parameters
    ----------
    author_names : list[str]
        Author names to fetch.
    cache_path : Path or str
        Path to the JSON cache file.
    token : str
        ADS API token.
    max_papers : int
        Maximum papers per author.

    Returns
    -------
    dict
        ``{author_name: {"abstracts": [...], ...}}``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d authors from cache", len(data))
    else:
        data = {}

    to_fetch = [name for name in author_names if name not in data]
    logger.info("%d authors to fetch (%d cached)", len(to_fetch), len(data))

    for i, name in enumerate(to_fetch, 1):
        logger.info("[%d/%d] Fetching %s ...", i, len(to_fetch), name)
        try:
            result = fetch_author_papers(name, token, max_papers=max_papers)
            data[name] = result
        except Exception as e:
            logger.error("Error fetching %s: %s", name, e)
            data[name] = {"abstracts": [], "titles": [], "count": 0}

    # Save updated cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Cache saved to %s (%d total authors)", cache_path, len(data))

    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch author papers from ADS.")
    parser.add_argument(
        "--names", nargs="+", default=None,
        help='Author names in "Last, First" format',
    )
    parser.add_argument(
        "--output", type=str, default="output/reviewer_abstracts.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-papers", type=int, default=25,
        help="Max papers per author",
    )
    args = parser.parse_args()

    token = os.getenv("ADS_DEV_KEY", "")
    if not token:
        print("ERROR: Set the ADS_DEV_KEY environment variable.")
        return

    if args.names:
        author_names = args.names
    else:
        # Default: read from reviewers CSV if available
        try:
            import pandas as pd
            from src import config
            df = pd.read_csv(config.REVIEWERS_CSV)
            author_names = (
                df["last_name"].str.strip() + ", " + df["first_name"].str.strip()
            ).tolist()
            logger.info("Read %d authors from %s", len(author_names), config.REVIEWERS_CSV)
        except Exception:
            print("No --names provided and could not read reviewers CSV.")
            print("Usage: python scripts/fetch_ads_data.py --names 'Last, First' ...")
            return

    fetch_all(author_names, args.output, token, max_papers=args.max_papers)


if __name__ == "__main__":
    main()
