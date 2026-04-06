#!/usr/bin/env python3
"""
Analyze the discrepancy between TF-IDF and Transformer-based models.
Finds cases where one model succeeds and the other fails on the ADS dataset.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_loader import load_proposals, load_reviewers, load_ground_truth, load_reviewer_abstracts
from src.embeddings import TfidfEmbedder, SentenceTransformerEmbedder
from src.metrics import compute_ranking_metrics, get_top_matches

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze model discrepancy.")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR), help="Data directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    print(f"Loading data from {data_dir}...")
    
    proposals_df = load_proposals(data_dir / "proposals.csv")
    reviewers_df = load_reviewers(data_dir / "reviewers.csv")
    reviewer_abstracts = load_reviewer_abstracts(data_dir / "reviewer_abstracts.json")
    ground_truth = load_ground_truth(data_dir / "ground_truth.csv")
    
    # Prepare text corpora
    proposal_ids = proposals_df["proposal_id"].astype(str).tolist()
    proposal_texts = (proposals_df["title"].fillna("") + " " + proposals_df["abstract"].fillna("")).tolist()
    
    reviewer_ids = reviewers_df["reviewer_id"].astype(str).tolist()
    reviewer_names = reviewers_df["full_name"].tolist()
    reviewer_paper_lists = []
    for name in reviewer_names:
        data = reviewer_abstracts.get(name, {"abstracts": []})
        reviewer_paper_lists.append(data.get("abstracts", []))
        
    # Run TF-IDF
    tfidf = TfidfEmbedder()
    tfidf_scores = tfidf.compute_scores(proposal_texts, reviewer_paper_lists, proposal_ids, reviewer_ids)
    tfidf_metrics = compute_ranking_metrics(tfidf_scores, ground_truth, min_scores=1)
    
    # Run SentenceTransformer
    sbert = SentenceTransformerEmbedder()
    sbert_scores = sbert.compute_scores(proposal_texts, reviewer_paper_lists, proposal_ids, reviewer_ids)
    sbert_metrics = compute_ranking_metrics(sbert_scores, ground_truth, min_scores=1)
    
    # Compare
    comparison = []
    for pid in ground_truth.keys():
        if pid not in tfidf_metrics or pid not in sbert_metrics:
            continue
        
        comparison.append({
            "proposal_id": pid,
            "title": proposals_df[proposals_df["proposal_id"].astype(str) == pid]["title"].values[0],
            "tfidf_rank": tfidf_metrics[pid]["rank"],
            "sbert_rank": sbert_metrics[pid]["rank"],
            "diff": tfidf_metrics[pid]["rank"] - sbert_metrics[pid]["rank"]
        })
        
    comp_df = pd.DataFrame(comparison)
    
    # Top 5 Transformer Wins (SBERT better than TF-IDF)
    trans_wins = comp_df[comp_df["diff"] > 10].sort_values("diff", ascending=False).head(5)
    
    # Top 5 TF-IDF Wins (TF-IDF better than SBERT)
    tfidf_wins = comp_df[comp_df["diff"] < -10].sort_values("diff", ascending=True).head(5)
    
    print("\n" + "="*80)
    print("ANALYSIS: Transformer Wins (SBERT much better than TF-IDF)")
    print("="*80)
    for _, row in trans_wins.iterrows():
        pid = row["proposal_id"]
        true_author = ground_truth[pid]
        author_name = reviewers_df[reviewers_df["reviewer_id"].astype(str) == str(true_author)]["full_name"].values[0]
        
        print(f"\nPROPOSAL: {row['title']} (ID: {pid})")
        print(f"TRUE AUTHOR: {author_name} (ID: {true_author})")
        print(f"RANKS -> TF-IDF: {row['tfidf_rank']} | SBERT: {row['sbert_rank']}")
        
        print("\n  Top TF-IDF Recs:")
        for rec in get_top_matches(tfidf_scores, pid, reviewers_df, k=3):
            mark = " ✅" if rec["reviewer_id"] == str(true_author) else ""
            print(f"    - {rec['full_name']} (score: {rec['score']:.4f}){mark}")
            
        print("\n  Top SBERT Recs:")
        for rec in get_top_matches(sbert_scores, pid, reviewers_df, k=3):
            mark = " ✅" if rec["reviewer_id"] == str(true_author) else ""
            print(f"    - {rec['full_name']} (score: {rec['score']:.4f}){mark}")
            
    print("\n" + "="*80)
    print("ANALYSIS: TF-IDF Wins (TF-IDF much better than SBERT)")
    print("="*80)
    for _, row in tfidf_wins.iterrows():
        pid = row["proposal_id"]
        true_author = ground_truth[pid]
        author_name = reviewers_df[reviewers_df["reviewer_id"].astype(str) == str(true_author)]["full_name"].values[0]
        
        print(f"\nPROPOSAL: {row['title']} (ID: {pid})")
        print(f"TRUE AUTHOR: {author_name} (ID: {true_author})")
        print(f"RANKS -> TF-IDF: {row['tfidf_rank']} | SBERT: {row['sbert_rank']}")
        
        print("\n  Top TF-IDF Recs:")
        for rec in get_top_matches(tfidf_scores, pid, reviewers_df, k=3):
            mark = " ✅" if rec["reviewer_id"] == str(true_author) else ""
            print(f"    - {rec['full_name']} (score: {rec['score']:.4f}){mark}")
            
        print("\n  Top SBERT Recs:")
        for rec in get_top_matches(sbert_scores, pid, reviewers_df, k=3):
            mark = " ✅" if rec["reviewer_id"] == str(true_author) else ""
            print(f"    - {rec['full_name']} (score: {rec['score']:.4f}){mark}")

if __name__ == "__main__":
    main()
