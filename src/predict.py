"""Inference: load trained models and explain a prediction for one prompt."""
import string

import joblib
import numpy as np

from src.config import LABEL_NAMES, MODELS_DIR
from src.data import clean_text
from src.evaluate import load_results
from src.features import load_encoder

PUNCTUATION = string.punctuation

MODEL_LABELS = {
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_svm": "TF-IDF + SVM",
    "tfidf_knn": "TF-IDF + KNN",
    "tfidf_random_forest": "TF-IDF + Random Forest",
    "tfidf_xgboost": "TF-IDF + XGBoost",
    "emb_knn": "Embeddings + KNN",
    "emb_xgboost": "Embeddings + XGBoost",
}


class PromptDetector:
    """Loads saved models on demand and turns a raw prompt into a label plus explanations."""

    def __init__(self) -> None:
        self._models: dict = {}
        self._encoder = None
        self._cluster_model = None

    def available_models(self) -> list[str]:
        return [name for name in MODEL_LABELS if (MODELS_DIR / f"{name}.joblib").exists()]

    def _model(self, name: str):
        if name not in self._models:
            self._models[name] = joblib.load(MODELS_DIR / f"{name}.joblib")
        return self._models[name]

    def injection_probability(self, name: str, texts: list[str]) -> np.ndarray:
        """P(injection) for each already-cleaned text."""
        model = self._model(name)
        if name.startswith("emb_"):
            self._encoder = self._encoder or load_encoder()
            texts = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return model.predict_proba(texts)[:, 1]

    def top_words(self, name: str, text: str, top_n: int = 8) -> list[tuple[str, float]]:
        """Occlusion importance: change in log-odds of Injection when a word is removed.

        Model-agnostic, so it works identically for every model. Log-odds (not raw probability)
        keeps the signal visible even when the model is ~100% sure. Positive = towards Injection.
        """
        words = text.split()
        unique = list(dict.fromkeys(words))[:80]  # cap keeps long prompts fast
        variants = [" ".join(w for w in words if w != u) or " " for u in unique]
        probs = np.clip(self.injection_probability(name, [text] + variants), 1e-12, 1 - 1e-12)
        logit = np.log(probs / (1 - probs))
        impact = logit[0] - logit[1:]
        order = np.argsort(-np.abs(impact))[:top_n]
        return [(unique[i].strip(PUNCTUATION), float(impact[i])) for i in order]

    def attack_type(self, text: str) -> dict:
        """Nearest TF-IDF K-means cluster, described by its most characteristic terms."""
        path = MODELS_DIR / "tfidf_kmeans.joblib"
        if not path.exists():
            return {}
        self._cluster_model = self._cluster_model or joblib.load(path)
        vec = self._cluster_model["vectorizer"].transform([text])
        cluster = int(self._cluster_model["kmeans"].predict(vec)[0])
        return next(c for c in load_results()["tfidf_kmeans"]["clusters"] if c["cluster"] == cluster)

    def analyze(self, raw_text: str, name: str, top_n: int = 8) -> dict:
        text = clean_text(raw_text)
        p_inj = float(self.injection_probability(name, [text])[0])
        label = int(p_inj >= 0.5)  # default threshold, same as in evaluation
        return {
            "label": LABEL_NAMES[label],
            "confidence": p_inj if label else 1 - p_inj,
            "injection_probability": p_inj,
            "top_words": self.top_words(name, text, top_n),
        }
