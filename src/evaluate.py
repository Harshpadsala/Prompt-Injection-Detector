"""Metrics, confusion-matrix plots and the shared results file."""
import json

import matplotlib

matplotlib.use("Agg")  # save figures without a display (runs on a server)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.config import FIGURES_DIR, LABEL_NAMES, REPORTS_DIR

RESULTS_PATH = REPORTS_DIR / "metrics.json"


def get_scores(model, X) -> np.ndarray:
    """Injection-class score in [0, 1] (or a margin) for ranking metrics like AUROC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, scores) -> dict:
    """Metrics for the injection class (label 1). Recall matters most: a missed attack is costly."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, scores),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=list(LABEL_NAMES.values()), ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def evaluate_model(model, X, y, split: str, run_name: str) -> dict:
    """Score a fitted model on one split, save its confusion matrix, return the metrics."""
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, get_scores(model, X))
    plot_confusion_matrix(y, y_pred, f"{run_name} ({split})", f"cm_{run_name}_{split}.png")
    return metrics


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}


def record_result(run_name: str, payload: dict) -> None:
    """Merge one run's results into reports/metrics.json under its run name."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    results[run_name] = {**results.get(run_name, {}), **payload}
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float))


def plot_k_sweep(rows: list[dict], filename: str) -> None:
    """Elbow (inertia) and silhouette curves used to choose K."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(ks, [r["inertia"] for r in rows], marker="o")
    axes[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
    axes[1].plot(ks, [r["silhouette"] for r in rows], marker="o", color="tab:orange")
    axes[1].set(title="Silhouette score", xlabel="K", ylabel="Silhouette")
    for ax in axes:
        ax.set_xticks(range(min(ks) + min(ks) % 2, max(ks) + 1, 2))  # even K values only
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auroc"]


def metrics_table(results: dict, split: str, labels: dict[str, str]) -> pd.DataFrame:
    """Supervised models x metrics for one split, best recall first. labels maps run name to display title."""
    rows = {labels[n]: r[split] for n, r in results.items() if n in labels and split in r}
    return pd.DataFrame(rows).T[METRIC_COLUMNS].sort_values("recall", ascending=False)
