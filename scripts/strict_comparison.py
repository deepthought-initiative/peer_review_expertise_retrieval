import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings import (
    KeywordEmbedder, TfidfEmbedder, LdaEmbedder, 
    SpecterEmbedder, SentenceTransformerEmbedder, Gpt4oEmbedder
)
from src.metrics import compute_ranking_metrics, compute_ndcg_per_proposal
from src import config

PROJECT = config.PROJECT_ROOT
DATA = PROJECT / "original_data"

def load_all_data():
    """Matches the data loading logic from the research notebook/scripts."""
    if not DATA.exists():
        print(f"\n[ERROR] Proprietary data directory not found at: {DATA}")
        print("This script requires the original ESO dataset to run side-by-side parity checks.")
        print("To run the simulation-only workflow, use 'scripts/run_demo.py' instead.\n")
        sys.exit(1)
        
    # 1. Ground Truth
    with open(DATA / "DPR_input_P110.json") as f:
        input_data = json.load(f)
    map_proposal_to_reviewer = {}
    for p in input_data["proposalsForReviewDTO"]["proposals"]:
        pid = p.get("proposalId")
        for inv in p.get("investigators", []):
            if inv.get("isReviewer"):
                map_proposal_to_reviewer[pid] = str(inv.get("accountId"))
                break
    gt = {str(k): str(v) for k,v in map_proposal_to_reviewer.items()}
    print(f"[DEBUG] Initial Ground Truth: {len(gt)} proposals")

    # 2. Proposals
    pis_df = pd.read_csv(DATA / "ESO_DPR_PIs_105-113.csv", sep="|")
    pis_p110 = pis_df[pis_df["name"] == "DPR P110"].copy()
    pis_p110["pid_str"] = pis_p110["phase1_proposal_id"].astype(str)
    
    p_texts, p_ids = [], []
    missing_from_pis = []
    for pid in sorted(gt.keys()):
        row = pis_p110[pis_p110["pid_str"] == pid]
        if not row.empty:
            text = str(row["abstract"].iloc[0])
            p_texts.append(text)
            p_ids.append(pid)
        else:
            missing_from_pis.append(pid)
            
    print(f"[DEBUG] After filtering by PIs CSV: {len(p_ids)} proposals")
    if missing_from_pis:
        print(f"[DEBUG] Missing from PIs CSV ({len(missing_from_pis)}): {missing_from_pis[:10]}...")

    # 3. Reviewers & Abstracts
    rev_df = pd.read_csv(DATA / "DPR_reviewers_names_110_113.csv", sep=";")
    p110_rev = rev_df[rev_df["name"] == "DPR P110"].drop_duplicates(subset="account_id").copy()
    p110_rev["fname"] = p110_rev["last_name"].str.strip() + ", " + p110_rev["first_name"].str.strip()
    id_to_name = dict(zip(p110_rev["account_id"], p110_rev["fname"]))
    name_to_id = dict(zip(p110_rev["fname"], p110_rev["account_id"]))
    
    from src.preprocessing import TYPO_MAP, REVERSE_TYPO_MAP
    
    ads_df = pd.read_json(DATA / "eso_ads25_1_22_26.json")
    def unique_list(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]
    
    # Check for missing reviewers in id_to_name
    missing_rev_ids = []
    mapped_names = []
    valid_p_ids = []
    valid_p_texts = []
    
    for pid, text in zip(p_ids, p_texts):
        rev_id = int(gt[pid])
        if rev_id in id_to_name:
            raw_name = id_to_name[rev_id]
            mapped_name = TYPO_MAP.get(raw_name, raw_name)
            mapped_names.append(mapped_name)
            valid_p_ids.append(pid)
            valid_p_texts.append(text)
        else:
            missing_rev_ids.append(rev_id)
            
    print(f"[DEBUG] After checking Reviewer Names: {len(valid_p_ids)} proposals")
    if missing_rev_ids:
        print(f"[DEBUG] Missing Reviewer IDs in names CSV ({len(set(missing_rev_ids))} unique): {list(set(missing_rev_ids))[:10]}...")

    rev_list_ordered = unique_list(mapped_names)
    
    ads_df['fname_ordered'] = pd.Categorical(ads_df['fname'], categories=rev_list_ordered, ordered=True)
    ads_df_sorted = ads_df.sort_values('fname_ordered').dropna(subset=['fname_ordered']).reset_index(drop=True)
    
    print(f"[DEBUG] Found {len(ads_df_sorted['fname'].unique())} reviewers in ADS out of {len(rev_list_ordered)} expected.")
    if len(ads_df_sorted['fname'].unique()) < len(rev_list_ordered):
        missing = set(rev_list_ordered) - set(ads_df_sorted['fname'].unique())
        print(f"[DEBUG] Missing Reviewers from ADS ({len(missing)}): {list(missing)}")
    
    r_papers = ads_df_sorted["abstract"].tolist()
    # Map back to account_ids (Clean -> Typo)
    r_ids = [str(name_to_id.get(REVERSE_TYPO_MAP.get(n, n))) for n in ads_df_sorted["fname"]]

    # 4. JSON-based Expertise (for NDCG and Conflicts)
    # Replicating user provided snippet exactly
    with open(DATA / 'DPR_output_matrix_P110.json', 'r') as f:
        matrix_data = json.load(f)
    
    account_ids, prp_ids, expertise_scores, conflicts = [], [], [], []
    for proposal in matrix_data['Data']['proposalAssignment']:
        pid = proposal['proposalId']
        for reviewer in proposal['reviewerAssigned']:
            account_ids.append(reviewer['accountId'])
            prp_ids.append(pid)
            expertise_scores.append(reviewer['assignmentInfo']['score'])
            conflicts.append(len(reviewer['assignmentInfo']['conflicts']) > 0)
    
    json_df = pd.DataFrame({
        'account_id': [str(a) for a in account_ids],
        'proposal_id': [str(p) for p in prp_ids],
        'expertise_score': expertise_scores,
        'conflict': conflicts
    }).drop_duplicates(subset=['account_id', 'proposal_id'])
    
    # Universal Pool (matches notebook baseline candidate set)
    all_system_reviewers = sorted(json_df['account_id'].unique())
    print(f"[DEBUG] Universal Pool from JSON Matrix: {len(all_system_reviewers)} reviewers")

    # Merge with expert labels
    exp_df = pd.read_csv(DATA / 'DPR_reviewers_expertise_110_113.csv', sep=';')
    exp_df.rename(columns={'phase1_proposal_id': 'proposal_id'}, inplace=True)
    exp_df['proposal_id'] = exp_df['proposal_id'].astype(str)
    exp_df['account_id'] = exp_df['account_id'].astype(str)
    
    final_data = json_df.merge(
        exp_df[['account_id', 'proposal_id', 'expertise_state_id']],
        on=['account_id', 'proposal_id'],
        how='inner'
    )
    final_data = final_data.drop_duplicates(subset=['account_id', 'proposal_id'])
    
    # Final filter for NDCG
    # Replicating notebook filtering exactly:
    # (~final_data['expertise_state_id'].isin(['(null)', '1']))
    # Note: If it's an int64, this filter won't catch int 1, so it stays in.
    final_ndcg_df = final_data[
        (final_data['conflict'] == False) &
        (~final_data['expertise_state_id'].isin(['(null)', '1']))
    ].copy()
    
    # 5. Map to NDCG labels
    # If state 1 is present, map it to 0 (Non-Expert) unless otherwise specified.
    # Our RELEVANCE_MAP already handles 2, 3, 4. We'll add 1 -> 0 explicitly.
    final_ndcg_df['expertise_label'] = final_ndcg_df['expertise_state_id'].astype(int)
    
    print(f"[DEBUG] NDCG Pool size: {len(final_ndcg_df)} pairs across {final_ndcg_df['proposal_id'].nunique()} proposals.")

    return valid_p_texts, r_papers, valid_p_ids, r_ids, gt, final_ndcg_df, name_to_id, all_system_reviewers, json_df

def run_comparison():
    from src.preprocessing import TYPO_MAP, REVERSE_TYPO_MAP
    
    p_texts, r_papers, p_ids, r_ids, gt, exp_df, name_to_id, all_revs, raw_json_df = load_all_data()
    
    # Pre-build "Original Keywords" matrix from JSON (expertise_score / 2)
    keywords_original_pivot = raw_json_df.pivot(index='account_id', columns='proposal_id', values='expertise_score') / 2.0
    # Reindex to all proposals and all reviewers
    keywords_original_matrix = keywords_original_pivot.reindex(index=all_revs, columns=p_ids).fillna(0)

    methods_config = {
        'Keywords': KeywordEmbedder(DATA / "shared_proposal_reviewer_full_matrix.csv"),
        'TF-IDF': TfidfEmbedder(),
        'LDA': LdaEmbedder(),
        'SPECTER2': SpecterEmbedder(),
        'SentenceTransformer': SentenceTransformerEmbedder(),
        'GPT-4o mini': Gpt4oEmbedder(DATA / "reviewer_matches_COMPLETE.csv")
    }
    
    requested_methods = ['Keywords', 'LDA', 'TF-IDF', 'SPECTER2', 'SentenceTransformer', 'GPT-4o mini']
    results = []

    for source_label in ['Original', 'Refactored']:
        print(f"\n  - Running {source_label} pipeline...")
        for method_name in requested_methods:
            embedder = methods_config[method_name]
            
            # 1. Scores (Matrix Reindexing to Universal Pool)
            if method_name == 'Keywords' and source_label == 'Original':
                scores = keywords_original_matrix.copy()
            else:
                raw_scores = embedder.compute_scores(p_texts, r_papers, p_ids, r_ids)
                
                # Map GPT names back to account_ids if needed
                if method_name == 'GPT-4o mini':
                    new_idx = []
                    for n in raw_scores.index:
                        clean_n = n.strip('"')
                        typo_n = REVERSE_TYPO_MAP.get(clean_n, clean_n)
                        acc_id = name_to_id.get(typo_n)
                        new_idx.append(str(acc_id))
                    raw_scores.index = new_idx
                    raw_scores = raw_scores[raw_scores.index != 'None']
                
                # REINDEX to the universal pool (435 reviewers) with 0 as fallback
                # This ensures NDCG matches the research baseline exactly.
                scores = raw_scores.reindex(index=all_revs).fillna(0)
            
            # 2. Ranking Metrics (MRR, Rank, Hit@25, Z)
            # Match the set of valid items (N)
            ranking = compute_ranking_metrics(scores, gt, [25], 10)
            valid_n = len(ranking)
            print(f"    [{method_name}] Processed {valid_n} valid items.")
            
            # 3. NDCG (uses the exact final_ndcg_df pool)
            # Filter scores to only include items in the NDCG pool for consistency
            # However, compute_ndcg_per_proposal internally takes long_df
            # We must melt 'scores' and merge with 'exp_df' (which is now final_ndcg_df)
            mat_copy = scores.copy()
            mat_copy.index.name = 'account_id'
            long_df = mat_copy.reset_index().melt(id_vars='account_id', var_name='proposal_id', value_name='score')
            long_df['account_id'] = long_df['account_id'].astype(str)
            long_df['proposal_id'] = long_df['proposal_id'].astype(str)

            # Join to the PRE-FILTERED final_ndcg_df (which already has conflict=False and labels)
            long_ndcg = long_df.merge(
                exp_df[['account_id', 'proposal_id', 'expertise_label']],
                on=['account_id', 'proposal_id'],
                how='inner' # Maintain exact match with notebook's pool
            )
            long_ndcg['score'] = long_ndcg['score'].fillna(0)
            ndcg_results = compute_ndcg_per_proposal(long_ndcg, score_column='score', relevance_column='expertise_label')
            ndcg_dict = dict(zip(ndcg_results['proposal_id'].astype(str), ndcg_results['ndcg']))
            
            # Aggregation logic
            mrr_vals = [m['mrr'] for m in ranking.values()]
            rank_vals = [m['rank'] for m in ranking.values()]
            hit_vals = [m['hit@25'] for m in ranking.values()]
            z_vals = [m['z'] for m in ranking.values() if not np.isnan(m['z'])]
            ndcg_vals = [v for v in ndcg_dict.values() if not np.isnan(v)]
            
            results.append({
                'Method': method_name,
                'Source': source_label,
                'N': valid_n,
                'MRR': np.mean(mrr_vals),
                'Median Rank': np.median(rank_vals),
                'Hit@25': np.mean(hit_vals),
                'Z-score': np.mean(z_vals) if z_vals else 0.0,
                'NDCG': np.mean(ndcg_vals) if ndcg_vals else 0.0
            })
            
    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by=['Method', 'Source'])
    
    # Save to CSV
    cols = ['Method', 'Source', 'N', 'Median Rank', 'MRR', 'Hit@25', 'Z-score', 'NDCG']
    output_path = PROJECT / "output" / "strict_comparison.csv"
    os.makedirs(PROJECT / "output", exist_ok=True)
    final_df[cols].to_csv(output_path, index=False)
    
    print("\n" + "="*105)
    print("STRICT SIDE-BY-SIDE COMPARISON (NOTEBOOK VS. SRC/)")
    print("="*105)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    print(final_df[cols].to_string(index=False))
    print("="*105)
    print(f"✅ Results saved to: {output_path}\n")

if __name__ == "__main__":
    run_comparison()
