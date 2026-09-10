import numpy as np
import pandas as pd
from numpy.typing import NDArray
from itertools import combinations
from collections.abc import Sequence

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)


def evaluate_kmeans_candidates(features: NDArray[np.float64], min_clusters: int=2, max_clusters: int=10, random_state: int=0, silhouette_sample_size: int=5000) -> pd.DataFrame:
    """Evaluate K-means solutions across a range of cluster counts."""
    results: list[dict[str, int | float]]= []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(features)
        sample_size = min(silhouette_sample_size, len(features))
        silhouette = silhouette_score(features, labels, sample_size=sample_size, random_state=random_state)
        davies_bouldin = (davies_bouldin_score(features, labels))
        calinski_harabasz = (calinski_harabasz_score(features, labels))
        results.append({ "n_clusters": n_clusters, "inertia": kmeans.inertia_,
                         "silhouette_score": silhouette, "davies_bouldin_score": davies_bouldin,
                         "calinski_harabasz_score": calinski_harabasz})

    return pd.DataFrame(results)

def evaluate_kmeans_stability(features: NDArray[np.float64], n_clusters: int, random_states: Sequence[int]=(0, 1, 2, 3, 4)) -> dict[str, float]:
    """Measure K-means stability across different random seeds."""
    label_sets: list[NDArray[np.int_]] = []
    for random_state in random_states:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(features)
        label_sets.append(labels)

    ari_scores: list[float] = []
    for labels_a, labels_b in combinations(label_sets, 2):
        ari_scores.append(adjusted_rand_score(labels_a, labels_b))

    return { "mean_ari": float(np.mean(ari_scores)), "min_ari": float(np.min(ari_scores)), "max_ari": float(np.max(ari_scores))}

def generate_candidate_cluster_labels(features: NDArray[np.float64], cluster_counts: Sequence[int]=(2, 3, 4), random_state: int=0) -> dict[int, NDArray[np.int_]]:
    """Generate K-means labels for spatial comparison."""
    labels_by_k: dict[int, NDArray[np.int_]] = {}
    for n_clusters in cluster_counts:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=random_state)
        labels_by_k[n_clusters] = (kmeans.fit_predict(features))

    return labels_by_k
