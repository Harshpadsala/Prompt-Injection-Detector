"""K-means helpers: choosing K, scoring clusters against true labels, describing clusters."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import SEED


def fit_kmeans(X, k: int) -> KMeans:
    # 3 restarts: stable results while staying fast on 261k prompts
    return KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(X)


def sweep_k(X, k_values, silhouette_sample: int = 10_000) -> list[dict]:
    """Inertia (elbow) and silhouette for each K. Silhouette uses a subsample to stay fast."""
    rows = []
    for k in k_values:
        km = fit_kmeans(X, k)
        sil = silhouette_score(
            X, km.labels_, sample_size=min(silhouette_sample, X.shape[0]), random_state=SEED
        )
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
        print(f"k={k:2d} inertia={km.inertia_:,.0f} silhouette={sil:.4f}")
    return rows


def cluster_purity(labels_true, labels_cluster) -> float:
    """Share of points that belong to the majority true class of their cluster."""
    labels_true, labels_cluster = np.asarray(labels_true), np.asarray(labels_cluster)
    correct = sum(
        np.bincount(labels_true[labels_cluster == c]).max() for c in np.unique(labels_cluster)
    )
    return float(correct / len(labels_true))


def score_clusters(labels_true, labels_cluster) -> dict:
    return {
        "purity": cluster_purity(labels_true, labels_cluster),
        "adjusted_rand_index": float(adjusted_rand_score(labels_true, labels_cluster)),
    }


def describe_clusters(cluster_labels, labels_true, term_weights, feature_names,
                      top_n: int = 8) -> list[dict]:
    """Per cluster: size, injection share, and the highest-weight terms.

    term_weights has shape (n_clusters, n_terms): each centroid expressed in term space.
    """
    labels_true, cluster_labels = np.asarray(labels_true), np.asarray(cluster_labels)
    info = []
    for c in range(term_weights.shape[0]):
        members = labels_true[cluster_labels == c]
        top_terms = [feature_names[i] for i in np.argsort(term_weights[c])[::-1][:top_n]]
        info.append({
            "cluster": c,
            "size": int(len(members)),
            "injection_share": float(members.mean()),
            "top_terms": top_terms,
        })
    return info
