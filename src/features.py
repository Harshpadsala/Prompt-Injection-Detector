"""Feature builders: TF-IDF vectorizer and sentence embeddings."""
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from src.config import EMBEDDING_MODEL, LSA_COMPONENTS, PROCESSED_DIR, SEED, TFIDF_MAX_FEATURES


def build_tfidf() -> TfidfVectorizer:
    """Unigram + bigram TF-IDF. Always used inside a Pipeline so it is fitted on train data only."""
    return TfidfVectorizer(
        # bigrams keep phrases like "ignore previous"; min_df=2 drops one-off typos;
        # stop words are kept because attack phrases are built from them ("ignore all")
        max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )


def build_lsa() -> Pipeline:
    """TF-IDF reduced to dense LSA vectors. K-means needs this: raw TF-IDF is too sparse and
    high-dimensional for distances to separate clusters (silhouette stays ~0)."""
    return Pipeline([
        ("tfidf", build_tfidf()),
        ("svd", TruncatedSVD(LSA_COMPONENTS, random_state=SEED)),
        ("norm", Normalizer(copy=False)),
    ])


@lru_cache(maxsize=1)
def load_encoder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[features] encoding on {device}")
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def embed_texts(texts: list[str], encoder: SentenceTransformer | None = None) -> np.ndarray:
    """Encode texts into L2-normalised sentence embeddings (n_texts x 384)."""
    encoder = encoder or load_encoder()
    return encoder.encode(
        # normalised vectors make cosine similarity a plain dot product
        texts, batch_size=512, normalize_embeddings=True, show_progress_bar=False
    )


def get_split_embeddings(name: str, texts: list[str]) -> np.ndarray:
    """Embed a split once and cache it on disk; later calls load the cached array."""
    path = PROCESSED_DIR / f"emb_{name}.npy"
    if path.exists():
        return np.load(path)
    embeddings = embed_texts(texts)
    np.save(path, embeddings)
    return embeddings
