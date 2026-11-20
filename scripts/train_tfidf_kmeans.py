"""Approach 2: K-means clustering on TF-IDF reduced with LSA."""
import joblib
import numpy as np

from src.clustering import describe_clusters, fit_kmeans, score_clusters, sweep_k
from src.config import K_RANGE, MODELS_DIR, SEED, SWEEP_SAMPLE
from src.data import load_split
from src.evaluate import plot_k_sweep, record_result
from src.features import build_lsa


def main() -> None:
    train, test = load_split("train"), load_split("test")
    lsa = build_lsa()
    X_train = lsa.fit_transform(train["text_clean"])

    # Choose K on a subsample (fast), then fit the final model on all training data.
    idx = np.random.default_rng(SEED).permutation(X_train.shape[0])[:SWEEP_SAMPLE]
    rows = sweep_k(X_train[idx], K_RANGE)
    plot_k_sweep(rows, "kmeans_tfidf_k_sweep.png")
    # The elbow curve has no clear bend, so the silhouette peak decides K.
    best_k = max(rows, key=lambda r: r["silhouette"])["k"]
    print(f"chosen K (best silhouette) = {best_k}")

    km = fit_kmeans(X_train, best_k)
    X_test = lsa.transform(test["text_clean"])
    result = {
        "k": best_k,
        "k_sweep": rows,
        "train": score_clusters(train["label"], km.labels_),
        "test": score_clusters(test["label"], km.predict(X_test)),
        # Map LSA centroids back to term space so each cluster can be described by words.
        "clusters": describe_clusters(
            km.labels_, train["label"], km.cluster_centers_ @ lsa["svd"].components_,
            lsa["tfidf"].get_feature_names_out()),
    }
    print(result["train"], result["test"])
    record_result("tfidf_kmeans", result)
    # Saved together so the app can place a new prompt in its nearest cluster.
    joblib.dump({"vectorizer": lsa, "kmeans": km}, MODELS_DIR / "tfidf_kmeans.joblib")


if __name__ == "__main__":
    main()
