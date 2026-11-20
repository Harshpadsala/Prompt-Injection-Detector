"""Approach 3: KNN, XGBoost and K-means on sentence embeddings."""
import argparse

import joblib
import numpy as np

from src.clustering import fit_kmeans, score_clusters, sweep_k
from src.config import K_RANGE, MODELS_DIR, SEED, SWEEP_SAMPLE
from src.data import load_split
from src.evaluate import evaluate_model, plot_k_sweep, record_result
from src.features import get_split_embeddings
from src.models import embedding_specs, tune_and_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-kmeans", action="store_true", help="only train the classifiers")
    args = parser.parse_args()

    splits = {s: load_split(s) for s in ("train", "test", "external")}
    emb = {s: get_split_embeddings(s, df["text_clean"].tolist()) for s, df in splits.items()}
    y = {s: df["label"].to_numpy() for s, df in splits.items()}

    for spec in embedding_specs():
        model = tune_and_fit(spec, emb["train"], y["train"])
        run_name = f"emb_{spec.name}"
        joblib.dump(model, MODELS_DIR / f"{run_name}.joblib")
        metrics = {
            "best_params": spec.best_params,
            "test": evaluate_model(model, emb["test"], y["test"], "test", run_name),
            "external": evaluate_model(
                model, emb["external"], y["external"], "external", run_name),
        }
        record_result(run_name, metrics)
        print(f"[{spec.name}] test={metrics['test']}\n[{spec.name}] external={metrics['external']}")

    if args.skip_kmeans:
        return

    # K-means on embeddings: choose K on a subsample, fit on all training data.
    idx = np.random.default_rng(SEED).permutation(len(y["train"]))[:SWEEP_SAMPLE]
    rows = sweep_k(emb["train"][idx], K_RANGE)
    plot_k_sweep(rows, "kmeans_emb_k_sweep.png")
    best_k = max(rows, key=lambda r: r["silhouette"])["k"]  # same rule as TF-IDF K-means
    km = fit_kmeans(emb["train"], best_k)
    record_result("emb_kmeans", {
        "k": best_k,
        "k_sweep": rows,
        "train": score_clusters(y["train"], km.labels_),
        "test": score_clusters(y["test"], km.predict(emb["test"])),
        "external": score_clusters(y["external"], km.predict(emb["external"])),
    })
    joblib.dump(km, MODELS_DIR / "emb_kmeans.joblib")
    print(f"emb_kmeans K={best_k}")


if __name__ == "__main__":
    main()
