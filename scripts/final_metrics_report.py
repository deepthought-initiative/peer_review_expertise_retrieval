#!/usr/bin/env python3
"""
Final Metrics Comparison Report (Comprehensive)

Compares all 6 expertise representation methods across both the 
original research logic and the refined src/ implementation.
Exports the final result to CSV for publication/review.
"""
import sys, json, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path("/Users/vicenteamado/Documents/expertise")
DATA = PROJECT / "original_data"
OUTPUT = PROJECT / "output"
sys.path.insert(0, str(PROJECT))

from src.embeddings import (
    KeywordEmbedder, Gpt4oEmbedder, TfidfEmbedder,
    LdaEmbedder, SpecterEmbedder, SentenceTransformerEmbedder
)
from src.metrics import compute_ranking_metrics, compute_ndcg_per_proposal

# ===================================================================
# 1. Data Loading (Corrected for P110 Research Dataset)
# ===================================================================
def load_all_data():
    # Ground Truth
    with open(DATA / "DPR_input_P110.json") as f:
        input_data = json.load(f)
    map_proposal_to_reviewer = {}
    for p in input_data["proposalsForReviewDTO"]["proposals"]:
        pid = p.get("proposalId")
        for inv in p.get("investigators", []):
            if inv.get("isReviewer"):
                map_proposal_to_reviewer[pid] = str(inv.get("accountId"))
                break
    gt_id = {str(k): str(v) for k,v in map_proposal_to_reviewer.items()}

    # Proposals (with Abstracts)
    pis_df = pd.read_csv(DATA / "ESO_DPR_PIs_105-113.csv", sep="|")
    pis_p110 = pis_df[pis_df["name"] == "DPR P110"].copy()
    pis_p110["pid_str"] = pis_p110["phase1_proposal_id"].astype(str)
    
    proposal_texts = []
    proposal_ids = []
    for pid in sorted(gt_id.keys()):
        row = pis_p110[pis_p110["pid_str"] == pid]
        if not row.empty:
            text = str(row["title"].iloc[0]) + " " + str(row["abstract"].iloc[0])
            proposal_texts.append(text)
            proposal_ids.append(pid)

    # Reviewers (with Papers)
    rev_df = pd.read_csv(DATA / "DPR_reviewers_names_110_113.csv", sep=";")
    p110_rev = rev_df[rev_df["name"] == "DPR P110"].drop_duplicates(subset="account_id").copy()
    p110_rev["fname"] = p110_rev["last_name"].str.strip() + ", " + p110_rev["first_name"].str.strip()
    id_to_name = dict(zip(p110_rev["account_id"], p110_rev["fname"]))
    name_to_id = dict(zip(p110_rev["fname"], p110_rev["account_id"]))
    
    ads_df = pd.read_json(DATA / "eso_ads25_1_22_26.json")
    def unique_list(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]
    rev_list_ordered = unique_list([id_to_name.get(int(gt_id[pid])) for pid in proposal_ids if int(gt_id[pid]) in id_to_name])
    ads_df['fname_ordered'] = pd.Categorical(ads_df['fname'], categories=rev_list_ordered, ordered=True)
    ads_df_sorted = ads_df.sort_values('fname_ordered').drop('fname_ordered', axis=1)
    ads_df_sorted.reset_index(drop=True, inplace=True)
    
    reviewer_papers = ads_df_sorted["abstract"].tolist()
    reviewer_ids = [str(name_to_id.get(n)) for n in ads_df_sorted["fname"]]

    # Expertise States for NDCG
    exp_df = pd.read_csv(DATA / 'DPR_reviewers_expertise_110_113.csv', sep=';')
    exp_df.rename(columns={'phase1_proposal_id': 'proposal_id'}, inplace=True)
    exp_df['proposal_id'] = exp_df['proposal_id'].astype(str)
    exp_df['account_id'] = exp_df['account_id'].astype(str)

    return proposal_texts, reviewer_papers, proposal_ids, reviewer_ids, gt_id, exp_df, name_to_id

# ===================================================================
# 2. Metric Calculation
# ===================================================================
def calculate_metrics(scores_df, gt_id, exp_df):
    # Ranking metrics (MRR, Z-Score, Hit@25, Rank)
    ranking = compute_ranking_metrics(scores_df, gt_id, [25], 10)
    z_vals = [m['z'] for m in ranking.values()]
    mrr_vals = [m['mrr'] for m in ranking.values()]
    h25_vals = [m['hit@25'] for m in ranking.values()]
    rank_vals = [m['rank'] for m in ranking.values()]
    
    # NDCG
    mat_copy = scores_df.copy()
    mat_copy.index.name = 'account_id'
    long_df = mat_copy.reset_index().melt(id_vars='account_id', var_name='proposal_id', value_name='score')
    long_df['account_id'] = long_df['account_id'].astype(str)
    long_df['proposal_id'] = long_df['proposal_id'].astype(str)
    
    long_ndcg = long_df.merge(exp_df[['account_id', 'proposal_id', 'expertise_state_id']], on=['account_id', 'proposal_id'], how='inner')
    long_ndcg = long_ndcg[~long_ndcg['expertise_state_id'].astype(str).isin(['nan', '(null)', '1'])]
    long_ndcg['expertise_label'] = long_ndcg['expertise_state_id'].astype(int) 
    
    ndcg_df = compute_ndcg_per_proposal(long_ndcg, score_column='score', relevance_column='expertise_label')
    ndcg_vals = ndcg_df['ndcg'].dropna().values
    
    return {
        "MRR": np.mean(mrr_vals),
        "Z-Score": np.mean(z_vals),
        "Hit@25": np.mean(h25_vals),
        "Median Rank": np.median(rank_vals),
        "NDCG": np.mean(ndcg_vals)
    }

# ===================================================================
# 3. Execution
# ===================================================================
if __name__ == "__main__":
    print("Loading datasets...")
    p_texts, r_papers, p_ids, r_ids, gt, exp, name_to_id = load_all_data()

    methods = {
        "Keywords": KeywordEmbedder(DATA / "shared_proposal_reviewer_full_matrix.csv"),
        "GPT-4o-mini": Gpt4oEmbedder(DATA / "reviewer_matches_COMPLETE.csv"),
        "TF-IDF": TfidfEmbedder(),
        "LDA": LdaEmbedder(),
        "SPECTER2": SpecterEmbedder(),
        "SentenceTransformer": SentenceTransformerEmbedder()
    }

    print("\nGenerating final comparison report...")
    rows = []
    for name, embedder in methods.items():
        print(f"Computing {name}...")
        scores = embedder.compute_scores(p_texts, r_papers, p_ids, r_ids)
        
        # Mapping: if index is names (GPT method), convert to account_ids
        if name == "GPT-4o-mini":
            # Map Reviewer names string to account_ids string
            # Some names might have quotes from CSV, strip them
            scores.index = [str(name_to_id.get(n.strip('"'))) for n in scores.index]
            # Filter out None if any names didn't match
            scores = scores[scores.index != 'None']

        m = calculate_metrics(scores, gt, exp)
        m["Method"] = name
        rows.append(m)

    # Reorder columns
    df = pd.DataFrame(rows)
    df = df[["Method", "MRR", "Z-Score", "Hit@25", "Median Rank", "NDCG"]]
    
    # Save to multiple formats
    csv_out = OUTPUT / "final_comparison_report.csv"
    json_out = OUTPUT / "final_comparison_report.json"
    
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient="records", indent=4)

    print(f"\nReport saved successfully to:\n- {csv_out}\n- {json_out}")
    print("\nFinal Results Table:")
    print(df.to_markdown(index=False))
