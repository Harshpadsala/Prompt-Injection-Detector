"""Model definitions, hyper-parameter grids and the shared tune-then-fit routine."""
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.config import CV_FOLDS, SEED, TUNING_SAMPLE_SIZE
from src.features import build_tfidf


@dataclass
class ModelSpec:
    """An estimator, its search grid, and the parameter that controls its internal threads."""
    name: str
    estimator: BaseEstimator
    grid: dict
    n_jobs_param: str | None = None
    best_params: dict = field(default_factory=dict)


def _classifiers(prefix: str = "") -> list[ModelSpec]:
    """Classifier specs; prefix is the pipeline step name ('clf__') or '' for a bare estimator."""
    p = prefix
    return [
        # max_iter raised: the default does not converge on 50k sparse features
        ModelSpec("logreg", LogisticRegression(max_iter=1000, random_state=SEED),
                  {f"{p}C": [0.1, 1, 10]}),
        # LinearSVC has no probabilities, so it is calibrated to produce a confidence score.
        ModelSpec("svm", CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3),
                  {f"{p}estimator__C": [0.1, 1, 10]}),
        # cosine suits text vectors; brute force because tree indexes do not work on sparse data
        ModelSpec("knn", KNeighborsClassifier(metric="cosine", algorithm="brute"),
                  {f"{p}n_neighbors": [3, 5, 11]}, n_jobs_param=f"{p}n_jobs"),
        ModelSpec("random_forest", RandomForestClassifier(random_state=SEED),
                  {f"{p}n_estimators": [100, 300], f"{p}max_depth": [None, 50]},
                  n_jobs_param=f"{p}n_jobs"),
        # n_jobs=1 while tuning: GridSearchCV already runs fits in parallel (refit uses all cores).
        ModelSpec("xgboost", XGBClassifier(tree_method="hist", n_jobs=1, random_state=SEED),
                  {f"{p}n_estimators": [200, 400], f"{p}max_depth": [6, 10]},
                  n_jobs_param=f"{p}n_jobs"),
    ]


def select(specs: list[ModelSpec], names: list[str] | None) -> list[ModelSpec]:
    """Keep only the named specs (all if names is None)."""
    return [s for s in specs if names is None or s.name in names]


def tfidf_specs() -> list[ModelSpec]:
    """Approach 1: every classifier wrapped in a TF-IDF pipeline (vectorizer fits on train only)."""
    specs = _classifiers(prefix="clf__")
    for spec in specs:
        spec.estimator = Pipeline([("tfidf", build_tfidf()), ("clf", spec.estimator)])
    return specs


def embedding_specs(names=("knn", "xgboost")) -> list[ModelSpec]:
    """Approach 3: classifiers applied directly to sentence embeddings."""
    return [s for s in _classifiers() if s.name in names]


def tune_and_fit(spec: ModelSpec, X, y) -> BaseEstimator:
    """Grid-search on a stratified subsample, then refit the best config on all of (X, y)."""
    y = np.asarray(y)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))[:TUNING_SAMPLE_SIZE]
    X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    start = time.time()
    search = GridSearchCV(
        # F1 balances missed attacks and false alarms
        spec.estimator, spec.grid, scoring="f1", n_jobs=-1,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
    )
    search.fit(X_sub, y[idx])
    spec.best_params = search.best_params_

    final = clone(spec.estimator).set_params(**search.best_params_)
    if spec.n_jobs_param:
        final.set_params(**{spec.n_jobs_param: -1})
    final.fit(X, y)
    print(f"[{spec.name}] best={search.best_params_} cv_f1={search.best_score_:.4f} "
          f"time={time.time() - start:.0f}s")
    return final
