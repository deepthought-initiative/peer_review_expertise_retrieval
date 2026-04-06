#!/usr/bin/env python3
"""
Dual JSON Results Generator

Generates two separate JSON reports:
1. output/notebook_results.json (Baseline logic: joined text, low n-grams)
2. output/src_results.json (Refined logic: paper averaging, high n-grams)

Each JSON includes aggregate metrics, per-proposal metrics, and top-3 matches.
"""
import sys, json, os
import numpy as np
import pandas as pd
from pathlib import Path
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
# Data Loading (P110 Research Dataset)
# ===================================================================
def load_all_data():
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

    exp_df = pd.read_csv(DATA / 'DPR_reviewers_expertise_110_113.csv', sep=';')
    exp_df.rename(columns={'phase1_proposal_id': 'proposal_id'}, inplace=True)
    exp_df['proposal_id'] = exp_df['proposal_id'].astype(str)
    exp_df['account_id'] = exp_df['account_id'].astype(str)

    return proposal_texts, reviewer_papers, proposal_ids, reviewer_ids, gt_id, exp_df, name_to_id, id_to_name

# ===================================================================
# Result Generation
# ===================================================================
def get_report_data(embedder, name, p_texts, r_papers, p_ids, r_ids, gt, exp, name_to_id, id_to_name):
    print(f"  - {name}...")
    scores = embedder.compute_scores(p_texts, r_papers, p_ids, r_ids)
    
    if name == "GPT-4o-mini":
        scores.index = [str(name_to_id.get(n.strip('"'))) for n in scores.index]
        scores = scores[scores.index != 'None']

    # Ranking Metrics
    ranking = compute_ranking_metrics(scores, gt, [25], 10)
    
    # NDCG
    mat_copy = scores.copy()
    mat_copy.index.name = 'account_id'
    long_df = mat_copy.reset_index().melt(id_vars='account_id', var_name='proposal_id', value_name='score')
    long_df['account_id'] = long_df['account_id'].astype(str)
    long_df['proposal_id'] = long_df['proposal_id'].astype(str)
    long_ndcg = long_df.merge(exp_df[['account_id', 'proposal_id', 'expertise_state_id']], on=['account_id', 'proposal_id'], how='inner')
    long_ndcg = long_ndcg[~long_ndcg['expertise_state_id'].astype(str).isin(['nan', '(null)', '1'])]
    long_ndcg['expertise_label'] = long_ndcg['expertise_state_id'].astype(int) 
    ndcg_results = compute_ndcg_per_proposal(long_ndcg, score_column='score', relevance_column='expertise_label')
    ndcg_dict = dict(zip(ndcg_results['proposal_id'].astype(str), ndcg_results['ndcg']))

    # Aggregate
    mrr_vals = [m['mrr'] for m in ranking.values()]
    ndcg_vals = [v for v in ndcg_dict.values() if not np.isnan(v)]
    
    # Top 3 matches
    top_matches = {}
    for pid in p_ids:
        if pid in scores.columns:
            top_3 = scores[pid].sort_values(ascending=False).head(3)
            matches = []
            for rid, score in top_3.items():
                try:
                    rid_int = int(float(rid)) if rid != "None" else -1
                    rev_name = id_to_name.get(rid_int, f"Unknown ({rid})")
                except (ValueError, TypeError):
                    rev_name = f"Unknown ({rid})"
                matches.append({"reviewer": rev_name, "score": float(score)})
            top_matches[pid] = matches

    return {
        "method": name,
        "aggregates": {
            "mrr": float(np.mean(mrr_vals)),
            "ndcg": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0
        },
        "per_proposal": {pid: {"mrr": ranking.get(pid, {}).get("mrr", 0.0), "ndcg": ndcg_dict.get(pid, 0.0)} for pid in p_ids},
        "top_results": top_matches
    }

if __name__ == "__main__":
    p_texts, r_papers, p_ids, r_ids, gt, exp_df, n2id, id2n = load_all_data()

    # 1. GENERATE NOTEBOOK (BASELINE) REPORT
    print("Generating NOTEBOOK (Baseline) results...")
    # Mocking baseline behavior: joined text, low n-grams
    notebook_methods = {
        "Keywords": KeywordEmbedder(DATA / "shared_proposal_reviewer_full_matrix.csv"),
        "TF-IDF": TfidfEmbedder(ngram_range=(1,1), fit_on_proposals_only=False),
        "LDA": LdaEmbedder(num_topics=20)
    }
    # For transformers in baseline, we lack an easy non-averaging way in current src/ 
    # so we'll just include the core ones for manual comparison of logic changes.
    notebook_rows = []
    for name, embedder in notebook_methods.items():
        notebook_rows.append(get_report_data(embedder, name, p_texts, r_papers, p_ids, r_ids, gt, exp_df, n2id, id2n))
    
    with open(OUTPUT / "notebook_results.json", "w") as f:
        json.dump(notebook_rows, f, indent=4)

    # 2. GENERATE SRC (REFINED) REPORT
    print("Generating SRC (Refined) results...")
    src_methods = {
        "Keywords": KeywordEmbedder(DATA / "shared_proposal_reviewer_full_matrix.csv"),
        "TF-IDF": TfidfEmbedder(),
        "LDA": LdaEmbedder(),
        "SPECTER2": SpecterEmbedder(),
        "SentenceTransformer": SentenceTransformerEmbedder()
    }
    src_rows = []
    for name, embedder in src_methods.items():
        src_rows.append(get_report_data(embedder, name, p_texts, r_papers, p_ids, r_ids, gt, exp_df, n2id, id2n))

    with open(OUTPUT / "src_results.json", "w") as f:
        json.dump(src_rows, f, indent=4)

    print(f"\nCreated:\n- {OUTPUT}/notebook_results.json\n- {OUTPUT}/src_results.json")
