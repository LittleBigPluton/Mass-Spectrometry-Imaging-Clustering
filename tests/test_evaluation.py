import numpy as np

from sklearn.datasets import make_blobs

from msi_clustering.evaluation import (
    evaluate_kmeans_candidates,
    evaluate_kmeans_stability,
    generate_candidate_cluster_labels
)


def create_features():
    features, _ = make_blobs(n_samples=90, centers=3, n_features=4, cluster_std=0.3, random_state=42)
    return features


def test_evaluate_kmeans_candidates():
    features = create_features()
    results = evaluate_kmeans_candidates(features, min_clusters=2, max_clusters=4, random_state=0, silhouette_sample_size=90)
    expected_columns = {"n_clusters", "inertia", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"}
    assert set(results.columns) == expected_columns
    assert results["n_clusters"].tolist() == [2, 3, 4]
    assert len(results) == 3
    assert np.isfinite(results["inertia"]).all()


def test_evaluate_kmeans_stability():
    features = create_features()
    results = evaluate_kmeans_stability(features, n_clusters=3, random_states=(0, 1, 2))
    assert set(results) == {"mean_ari", "min_ari", "max_ari"}
    assert (results["min_ari"] <= results["mean_ari"] <= results["max_ari"])
    assert -1.0 <= results["min_ari"] <= 1.0
    assert -1.0 <= results["max_ari"] <= 1.0


def test_generate_candidate_cluster_labels():
    features = create_features()
    labels_by_k = (generate_candidate_cluster_labels(features, cluster_counts=(2, 3, 4), random_state=0))
    assert set(labels_by_k) == {2, 3, 4}
    for n_clusters, labels in (labels_by_k.items()):
        assert len(labels) == len(features)
        assert (len(np.unique(labels)) == n_clusters)
