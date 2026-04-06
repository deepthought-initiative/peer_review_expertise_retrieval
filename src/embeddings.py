"""
Embedding methods for comparing reviewer expertise with proposal content.

Each method is implemented as a class inheriting from :class:`BaseEmbedder`.
The common interface guarantees:

- ``fit(proposal_texts, reviewer_texts)`` — learn vocabulary/model (no-op for
  pre-trained transformers).
- ``compute_scores(proposal_texts, reviewer_texts, proposal_ids, reviewer_ids)``
  — return a ``pd.DataFrame`` of shape *(reviewers × proposals)* with
  similarity scores.

Two special cases load pre-computed CSVs rather than computing embeddings
on-the-fly: :class:`KeywordEmbedder` and :class:`Gpt4oEmbedder`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.preprocessing import (
    clean_text,
    normalize_matrix,
    preprocess_for_lda,
)

logger = logging.getLogger(__name__)


def _set_seeds() -> None:
    """Set global random seeds for reproducibility."""
    np.random.seed(config.RANDOM_SEED)
    try:
        import torch
        torch.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)
    except ImportError:
        pass


# =========================================================================
# Base class
# =========================================================================
class BaseEmbedder(ABC):
    """Abstract base class for all embedding methods."""

    name: str = "base"

    @abstractmethod
    def compute_scores(
        self,
        proposal_texts: list[str],
        reviewer_papers: list[list[str]],
        proposal_ids: list[str],
        reviewer_ids: list[str],
    ) -> pd.DataFrame:
        """Return a (reviewers × proposals) similarity DataFrame.

        Parameters
        ----------
        proposal_texts : list[str]
            Preprocessed or raw proposal abstracts.
        reviewer_papers : list[list[str]]
            List where each entry is a list of paper abstracts for one reviewer.
        proposal_ids : list[str]
            Identifiers for columns.
        reviewer_ids : list[str]
            Identifiers for rows (index).

        Returns
        -------
        pd.DataFrame
            Rows = reviewers, columns = proposals, values = similarity.
        """


# =========================================================================
# Pre-computed CSV loaders
# =========================================================================
class KeywordEmbedder(BaseEmbedder):
    """Loads pre-computed keyword-overlap scores from CSV.

    The keyword matching was performed upstream (e.g. by the observatory's
    proposal system) and is not recomputed here.  This class simply wraps
    the CSV loading for API consistency.
    """

    name = "Keywords"

    def __init__(self, csv_path: Path | str | None = None):
        self.csv_path = Path(csv_path) if csv_path else config.KEYWORD_SCORES_CSV

    def compute_scores(self, proposal_texts=None, reviewer_papers=None,
                       proposal_ids=None, reviewer_ids=None) -> pd.DataFrame:
        """Load and return the pre-computed keyword scores."""
        from src.data_loader import load_precomputed_scores
        logger.info("Loading pre-computed keyword scores from %s", self.csv_path)
        return load_precomputed_scores(self.csv_path)


class Gpt4oEmbedder(BaseEmbedder):
    """Loads pre-computed GPT-4o-mini scores from CSV.

    These scores were generated via the OpenAI structured-output API,
    asking the model to rate reviewer–proposal compatibility.  Pre-computed
    scores are shipped with the package so that users do not need an API key.
    """

    name = "GPT-4o-mini"

    def __init__(self, csv_path: Path | str | None = None):
        self.csv_path = Path(csv_path) if csv_path else config.GPT4O_SCORES_CSV

    def compute_scores(self, proposal_texts=None, reviewer_papers=None,
                       proposal_ids=None, reviewer_ids=None) -> pd.DataFrame:
        """Load and return the pre-computed GPT-4o-mini scores."""
        from src.data_loader import load_precomputed_scores
        logger.info("Loading pre-computed GPT-4o-mini scores from %s", self.csv_path)
        return load_precomputed_scores(self.csv_path)


# =========================================================================
# TF-IDF
# =========================================================================
class TfidfEmbedder(BaseEmbedder):
    """TF-IDF vectorisation with cosine similarity.

    The vectoriser is fit on the **combined** corpus of proposals and
    reviewer abstracts so that the vocabulary is shared.  Cosine
    similarity is computed via normalised dot product.
    """

    name = "TF-IDF"

    def __init__(
        self, 
        max_features: int | None = None,
        ngram_range: tuple[int, int] | None = None,
        fit_on_proposals_only: bool = True
    ):
        self.max_features = max_features or config.TFIDF_MAX_FEATURES
        self.ngram_range = ngram_range or getattr(config, "TFIDF_NGRAM_RANGE", (1, 1))
        self.fit_on_proposals_only = fit_on_proposals_only

    def compute_scores(
        self,
        proposal_texts: list[str],
        reviewer_papers: list[list[str]],
        proposal_ids: list[str],
        reviewer_ids: list[str],
    ) -> pd.DataFrame:
        _set_seeds()
        logger.info("Computing TF-IDF scores (%s features, ngrams=%s)", self.max_features, self.ngram_range)

        # For TF-IDF we use raw abstracts (matching notebook v5)
        raw_reviewers = [" ".join(papers) for papers in reviewer_papers]

        # Fit strategy
        vectoriser = TfidfVectorizer(
            stop_words="english", 
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        if self.fit_on_proposals_only:
            vectoriser.fit(proposal_texts)
        else:
            vectoriser.fit(proposal_texts + raw_reviewers)

        prop_vecs = normalize_matrix(vectoriser.transform(proposal_texts))
        rev_vecs = normalize_matrix(vectoriser.transform(raw_reviewers))

        # Cosine similarity: reviewers × proposals
        sim = rev_vecs.dot(prop_vecs.T)
        if issparse(sim):
            sim = sim.toarray()

        return pd.DataFrame(sim, index=reviewer_ids, columns=proposal_ids)


# =========================================================================
# LDA
# =========================================================================
class LdaEmbedder(BaseEmbedder):
    """Latent Dirichlet Allocation topic model with cosine similarity.

    Topic distributions are treated as dense vectors and compared via
    normalised dot product.  ``random_state`` is set to ``RANDOM_SEED``
    for determinism.
    """

    name = "LDA"

    def __init__(
        self,
        num_topics: int | None = None,
        passes: int | None = None,
        workers: int | None = None,
    ):
        self.num_topics = num_topics or config.LDA_NUM_TOPICS
        self.passes = passes or config.LDA_PASSES
        self.workers = workers or config.LDA_WORKERS

    def _to_dense(self, corpus, model) -> np.ndarray:
        """Convert gensim topic distributions to a dense numpy array."""
        vectors = []
        for bow in corpus:
            topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
            vec = np.zeros(self.num_topics)
            for topic_id, prob in topic_dist:
                vec[topic_id] = prob
            vectors.append(vec)
        return np.array(vectors)

    def compute_scores(
        self,
        proposal_texts: list[str],
        reviewer_papers: list[list[str]],
        proposal_ids: list[str],
        reviewer_ids: list[str],
    ) -> pd.DataFrame:
        from gensim import corpora
        from gensim.models.ldamulticore import LdaMulticore

        _set_seeds()
        logger.info(
            "Computing LDA scores (K=%d, passes=%d)",
            self.num_topics, self.passes,
        )

        # Tokenise - join papers for LDA doc-level topics
        proc_proposals = preprocess_for_lda(proposal_texts)
        proc_reviewers = preprocess_for_lda([" ".join(p) for p in reviewer_papers])

        # Build dictionary from both corpora to ensure full coverage
        dictionary = corpora.Dictionary(proc_reviewers + proc_proposals)
        rev_corpus = [dictionary.doc2bow(doc) for doc in proc_reviewers]
        prop_corpus = [dictionary.doc2bow(doc) for doc in proc_proposals]

        # Train LDA — seeded for reproducibility
        model = LdaMulticore(
            rev_corpus + prop_corpus,
            num_topics=self.num_topics,
            id2word=dictionary,
            passes=self.passes,
            workers=self.workers,
            random_state=config.RANDOM_SEED,
        )

        # Dense topic vectors
        rev_dense = normalize_matrix(self._to_dense(rev_corpus, model))
        prop_dense = normalize_matrix(self._to_dense(prop_corpus, model))

        # Cosine similarity via dot product
        sim = rev_dense.dot(prop_dense.T)
        return pd.DataFrame(sim, index=reviewer_ids, columns=proposal_ids)


# =========================================================================
# SentenceTransformer
# =========================================================================
class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embeddings from a SentenceTransformer model.

    Uses ``sentence-transformers/all-distilroberta-v1`` by default.
    Similarity is cosine via ``sentence_transformers.util.pytorch_cos_sim``.
    """

    name = "SentenceTransformer"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.SBERT_MODEL_NAME

    def compute_scores(
        self,
        proposal_texts: list[str],
        reviewer_papers: list[list[str]],
        proposal_ids: list[str],
        reviewer_ids: list[str],
    ) -> pd.DataFrame:
        import torch
        from sentence_transformers import SentenceTransformer, util

        _set_seeds()
        
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("SentenceTransformer: using %s device", device)

        model = SentenceTransformer(self.model_name, device=device)

        # Encode proposals
        prop_embs = model.encode(proposal_texts, convert_to_numpy=True)
        prop_embs = prop_embs / np.linalg.norm(prop_embs, axis=1, keepdims=True)

        # Paper Averaging (matches notebook v5's best performance)
        prop_embs = model.encode(proposal_texts, convert_to_numpy=True)
        prop_embs = prop_embs / np.linalg.norm(prop_embs, axis=1, keepdims=True)

        # Flatten reviewer papers for efficient batch encoding
        all_papers = []
        author_boundaries = [0]
        for papers in reviewer_papers:
            if not papers:
                author_boundaries.append(author_boundaries[-1])
                continue
            all_papers.extend(papers)
            author_boundaries.append(len(all_papers))
        
        if all_papers:
            logger.info("Encoding %d papers for %d reviewers...", len(all_papers), len(reviewer_papers))
            all_embs = model.encode(all_papers, convert_to_numpy=True)
            
            # Fallback embedding for empty paper lists
            fallback_emb = model.encode([""], convert_to_numpy=True)[0]

            # Re-aggregate and average
            rev_embs_list = []
            for i in range(len(reviewer_papers)):
                start, end = author_boundaries[i], author_boundaries[i+1]
                if start == end:
                    rev_embs_list.append(fallback_emb)
                else:
                    rev_embs_list.append(np.mean(all_embs[start:end], axis=0))
            rev_embs = np.array(rev_embs_list)
        else:
            fallback_emb = model.encode([""], convert_to_numpy=True)[0]
            rev_embs = np.tile(fallback_emb, (len(reviewer_ids), 1))

        rev_embs = rev_embs / np.linalg.norm(rev_embs, axis=1, keepdims=True)

        sim = util.pytorch_cos_sim(
            torch.from_numpy(rev_embs).float(),
            torch.from_numpy(prop_embs).float(),
        ).numpy()

        return pd.DataFrame(sim, index=reviewer_ids, columns=proposal_ids)


# =========================================================================
# SPECTER2
# =========================================================================
class SpecterEmbedder(BaseEmbedder):
    """SPECTER2 scientific-document embeddings with adapter weights.

    Documents are encoded in batches to avoid GPU memory issues.
    The [CLS] token embedding is used as the document representation.
    """

    name = "SPECTER"

    def __init__(
        self,
        base_model: str | None = None,
        adapter: str | None = None,
        batch_size: int | None = None,
    ):
        self.base_model = base_model or config.SPECTER_BASE_MODEL
        self.adapter = adapter or config.SPECTER_ADAPTER
        self.batch_size = batch_size or config.SPECTER_BATCH_SIZE

    def _encode_batch(self, texts: list[str], tokenizer, model, device) -> np.ndarray:
        """Encode a batch of texts and return [CLS] embeddings."""
        import torch

        all_embs = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for idx, i in enumerate(range(0, len(texts), self.batch_size)):
            if idx % 50 == 0:
                logger.info("Encoding batch %d/%d...", idx + 1, n_batches)
            batch = texts[i: i + self.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            with torch.no_grad():
                output = model(**inputs)
            embs = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(embs)
        return np.vstack(all_embs)

    def compute_scores(
        self,
        proposal_texts: list[str],
        reviewer_papers: list[list[str]],
        proposal_ids: list[str],
        reviewer_ids: list[str],
    ) -> pd.DataFrame:
        import torch
        from sentence_transformers import util

        _set_seeds()
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info("SPECTER2: using %s device", device)

        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoAdapterModel.from_pretrained(self.base_model)
        model.load_adapter(
            self.adapter, source="hf", load_as="specter2", set_active=True
        )
        model.to(device)
        model.eval()

        # Paper Averaging (matches notebook v5's best performance)
        prop_embs = self._encode_batch(proposal_texts, tokenizer, model, device)
        prop_embs = prop_embs / np.linalg.norm(prop_embs, axis=1, keepdims=True)

        # Flatten reviewer papers for efficient batch encoding
        all_papers = []
        author_boundaries = [0]
        for papers in reviewer_papers:
            if not papers:
                author_boundaries.append(author_boundaries[-1])
                continue
            all_papers.extend(papers)
            author_boundaries.append(len(all_papers))

        if all_papers:
            logger.info("Encoding %d papers for %d reviewers...", len(all_papers), len(reviewer_papers))
            all_embs = self._encode_batch(all_papers, tokenizer, model, device)
            
            # Fallback embedding for empty paper lists
            fallback_emb = self._encode_batch([""], tokenizer, model, device)[0]

            # Re-aggregate and average
            rev_embs_list = []
            for i in range(len(reviewer_papers)):
                start, end = author_boundaries[i], author_boundaries[i+1]
                if start == end:
                    rev_embs_list.append(fallback_emb)
                else:
                    rev_embs_list.append(np.mean(all_embs[start:end], axis=0))
            rev_embs = np.array(rev_embs_list)
        else:
            fallback_emb = self._encode_batch([""], tokenizer, model, device)[0]
            rev_embs = np.tile(fallback_emb, (len(reviewer_ids), 1))
        rev_embs = rev_embs / np.linalg.norm(rev_embs, axis=1, keepdims=True)

        sim = util.pytorch_cos_sim(
            torch.from_numpy(rev_embs).float(),
            torch.from_numpy(prop_embs).float(),
        ).numpy()

        return pd.DataFrame(sim, index=reviewer_ids, columns=proposal_ids)


# =========================================================================
# Registry — convenient lookup by name
# =========================================================================
EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {
    "keywords": KeywordEmbedder,
    "tfidf": TfidfEmbedder,
    "lda": LdaEmbedder,
    "specter": SpecterEmbedder,
    "sentence_transformer": SentenceTransformerEmbedder,
    "gpt4o": Gpt4oEmbedder,
}


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """Instantiate an embedder by short name.

    Parameters
    ----------
    name : str
        One of ``"keywords"``, ``"tfidf"``, ``"lda"``, ``"specter"``,
        ``"sentence_transformer"``, ``"gpt4o"``.
    **kwargs
        Forwarded to the embedder constructor.

    Returns
    -------
    BaseEmbedder
        An instance of the requested embedder.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder '{name}'. Choose from: {list(EMBEDDER_REGISTRY)}"
        )
    return EMBEDDER_REGISTRY[key](**kwargs)
