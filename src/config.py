"""
Global configuration for the expertise evaluation pipeline.

All file paths, random seeds, API keys, and model hyperparameters are
centralised here. API keys are read exclusively from environment variables
— **no secrets are ever hardcoded**.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Random Seed — used by NumPy, PyTorch, gensim, and scikit-learn to ensure
# perfectly deterministic results across runs.
# ---------------------------------------------------------------------------
RANDOM_SEED = 11

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
# Default to the public demo dataset; override with DATA_DIR env var
# to point at confidential data on your own machine.
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "demo"))

PROPOSALS_CSV = DATA_DIR / "proposals.csv"
REVIEWERS_CSV = DATA_DIR / "reviewers.csv"
REVIEWER_ABSTRACTS_JSON = DATA_DIR / "reviewer_abstracts.json"
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"
KEYWORD_SCORES_CSV = DATA_DIR / "keyword_scores.csv"
GPT4O_SCORES_CSV = DATA_DIR / "gpt4o_scores.csv"

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# API Keys (environment variables only)
# ---------------------------------------------------------------------------
ADS_TOKEN = os.getenv("ADS_DEV_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------
# TF-IDF
TFIDF_MAX_FEATURES = None
TFIDF_NGRAM_RANGE = (1, 2)

# LDA (gensim)
LDA_NUM_TOPICS = 50
LDA_PASSES = 10
LDA_WORKERS = 1

# SentenceTransformer
SBERT_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"

# SPECTER2
SPECTER_BASE_MODEL = "allenai/specter2_base"
SPECTER_ADAPTER = "allenai/specter2"
SPECTER_BATCH_SIZE = 8  # avoids OOM on modest hardware

# ADS query defaults (used by dummy_data / fetch script)
ADS_MAX_PAPERS_PER_AUTHOR = 25
ADS_RECENT_YEARS = 25

# Metrics
HIT_AT_K = [25]
MIN_SCORES_PER_PROPOSAL = 10
