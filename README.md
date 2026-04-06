# Comparing Expertise Representation Methods for Reviewer–Proposal Matching in Astronomy


A modular, reproducible pipeline for evaluating how different text representations (e.g., TF-IDF, LDA, Transformers) identify expert reviewers for scientific proposals. This repository uses the **NASA/ADS** API to simulate a distributed peer review environment with real-world telescope proposals and publication histories.


## Core Simulation Strategy


To evaluate matching performance without proprietary data, this pipeline uses a "**Hold-out Simulation**":
1.  **Proposals**: Fetch actual telescope proposals from HST/JWST.
2.  **Reviewers**: The Principal Investigator (PI) of each proposal is designated as the "True Expert."
3.  **Expertise Corpus**: A 5-year publication history for that PI is fetched from ADS, **holding out** the original proposal abstract.
4.  **Evaluation**: We measure how well each method ranks the PI as the top match for their own proposal.


## Methods Evaluated


| Method | Type | Notes |
|---|---|---|
| **TF-IDF** | Sparse bag-of-words | Fit on proposal/corpus; unigrams + bigrams |
| **LDA** | Topic model | **50-topic** Latent Dirichlet Allocation (gensim) |
| **SPECTER2** | Transformer | Scientific-document embeddings (AllenAI) |
| **SentenceTransformer** | Transformer | `all-distilroberta-v1` |
| **Keywords** | Keyword overlay | Pre-computed overlap scores based on ESO implementation |
| **GPT-4o-mini** | LLM | Structured output scoring (pre-computed) |


## Quick Start (ADS Simulation)


The simulation requires a [NASA/ADS API key](https://ui.adsabs.harvard.edu/user/settings/token).


```bash
# 1. Install dependencies
# Optionally create a virtual environment
python -m venv venv
pip install -r requirements.txt


# 2. Set ADS API Key
export ADS_DEV_KEY="your-ads-api-key"


# 3. Run the HST/JWST simulation (N=300)
python scripts/run_demo.py --proposals --n-recent 300 --seed 42


# 4. View results
# Summary table will print to console and save to output/demo_results.csv
```


### CLI Arguments
- `--proposals`: Use actual HST/JWST proposal records (bibstem: `hst..prop` / `jwst.prop`).
- `--year YYYY`: Target a specific year for randomized publication benchmarks.
- `--seed N`: Set a random seed for reproducible sampling and initialization.
- `--methods LIST`: Choose specific methods to run (e.g. `--methods TF-IDF SPECTER2`).


## Reproducibility & Hardware


- **Determinism**: Random seeds are set globally (`RANDOM_SEED = 11`) across NumPy, PyTorch, and gensim to ensure numerical parity.
- **Data Policy**: The original_data (proprietary ESO data) is excluded from this repository. Users should follow the "Quick Start" to generate a simulated benchmark.


## License


This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
