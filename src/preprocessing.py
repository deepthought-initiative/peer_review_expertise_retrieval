"""
Text preprocessing utilities for the expertise evaluation pipeline.

Functions cover three distinct preprocessing needs:
1. General text cleaning (lowercasing, stopwords, lemmatisation).
2. Tokenisation for topic models (gensim-compatible).
3. Sparse-matrix row-normalisation for cosine similarity.
"""

import re
import logging

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize as sklearn_normalize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap — download corpora only if missing
# ---------------------------------------------------------------------------
for _resource in ("corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"):
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource.split("/")[-1], quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Map for correcting UTF-8/Latin-1 encoding artefacts in reviewer names
# as found in the ADS JSON cache vs the official CSVs.
TYPO_MAP = {
    'BÃ©zard, Bruno': 'Bezard, Bruno',
    'Gurpide Lasheras, Andres': 'Gurpide, Andres',
    'MartÃ\xadnez Aldama, Mary Loli': 'Martinez Aldama, Mary Loli',
    'PourrÃ©, Nicolas': 'Pourre, Nicolas',
    'Lafarga Magro, Marina': 'Lafarga, Marina',
    'AgÃ¼Ã\xad FernÃ¡ndez, JosÃ© Feliciano': 'Agüí Fernández, José Feliciano',
    'Blagorodnova Mujortova, Nadejda': 'Blagorodnova, Nadejda',
    'VillaseÃ±or, Jaime': 'Villasenor, Jaime',
    'Toloza Castillo, Odette': 'Toloza, Odette',
    'GonzÃ¡lez LÃ³pez, Jorge': 'Gonzalez Lopez, Jorge'
}

# Reverse map (Correct -> ADS Cache Key) used to align CSV names with JSON keys
REVERSE_TYPO_MAP = {v: k for k, v in TYPO_MAP.items()}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean and normalise a raw text string.

    Processing steps (order matters):
      1. Lowercase.
      2. Strip residual HTML tags (``<P />``, etc.).
      3. Remove non-alphabetic characters.
      4. Remove English stopwords.
      5. Lemmatise remaining tokens.
      6. Drop tokens shorter than 3 characters (removes noise like
         single-letter remnants of equations).

    Parameters
    ----------
    text : str
        Raw text to clean.

    Returns
    -------
    str
        Space-joined cleaned tokens.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove common HTML artefacts found in observatory proposal systems
    text = text.replace("<P />", "").replace("<p />", "")
    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()
    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 2
    ]
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# Tokenisation for gensim LDA
# ---------------------------------------------------------------------------
def preprocess_for_lda(documents: list[str]) -> list[list[str]]:
    """Tokenise documents into word lists suitable for gensim LDA.

    Uses ``gensim.utils.simple_preprocess`` (which lowercases, strips
    accents, and filters by token length) followed by stopword removal.

    Parameters
    ----------
    documents : list[str]
        Raw document strings.

    Returns
    -------
    list[list[str]]
        List of token lists, one per document.
    """
    return [
        [word for word in simple_preprocess(doc) if word not in STOPWORDS]
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# Matrix normalisation
# ---------------------------------------------------------------------------
def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation of a dense or sparse matrix.

    This is a prerequisite for computing cosine similarity via a simple
    dot product: ``cos(A, B) = norm(A) · norm(B)^T``.

    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse matrix
        Input matrix to normalise.

    Returns
    -------
    np.ndarray or scipy.sparse matrix
        Row-normalised matrix (same type as input).
    """
    if issparse(matrix):
        return sklearn_normalize(matrix, norm="l2", axis=1)
    # Dense path
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for all-zero rows
    norms[norms == 0] = 1.0
    return matrix / norms
