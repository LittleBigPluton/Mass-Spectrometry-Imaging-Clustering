import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)


def evaluate_kmeans_candidates(features, min_clusters=2, max_clusters=10, random_state=0, silhouette_sample_size=5000):
    """Evaluate K-means solutions across a range of cluster counts."""
    results = []
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

def evaluate_kmeans_stability(features, n_clusters, random_states=(0, 1, 2, 3, 4)):
    """Measure K-means stability across different random seeds."""
    label_sets = []
    for random_state in random_states:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(features)
        label_sets.append(labels)

    ari_scores = []
    for labels_a, labels_b in combinations(label_sets, 2):
        ari_scores.append(adjusted_rand_score(labels_a, labels_b))

    return { "mean_ari": float(np.mean(ari_scores)), "min_ari": float(np.min(ari_scores)), "max_ari": float(np.max(ari_scores))}

def generate_candidate_cluster_labels(features, cluster_counts=(2, 3, 4), random_state=0):
    """Generate K-means labels for spatial comparison."""
    labels_by_k = {}
    for n_clusters in cluster_counts:
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=random_state)
        labels_by_k[n_clusters] = (kmeans.fit_predict(features))

    return labels_by_k
