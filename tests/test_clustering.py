import numpy as np
import pandas as pd

import msi_clustering.clustering as clustering_module
from msi_clustering.clustering import MSIClusterer


def create_clusterer():
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(20, 3))
    cluster_b = rng.normal(loc=5.0, scale=0.2, size=(20, 3))
    features = np.vstack([cluster_a, cluster_b])
    clusterer = MSIClusterer("synthetic.csv")
    clusterer.data = pd.DataFrame({"X": np.arange(len(features)), "Y": np.zeros(len(features)), "metadata": np.zeros(len(features)),
                                   "mz_1": features[:, 0], "mz_2": features[:, 1], "mz_3": features[:, 2],})

    clusterer.mz_values = ["mz_1", "mz_2", "mz_3"]
    return clusterer


def test_apply_pca():
    clusterer = create_clusterer()
    clusterer.apply_pca(n_components=2)
    assert clusterer.pca_result.shape == (40, 2)
    assert len(clusterer.explained_variance_ratio_) == 2


def test_select_pca_components(tmp_path, monkeypatch):
    monkeypatch.setattr(clustering_module, "figures_dir", tmp_path)
    clusterer = create_clusterer()
    clusterer.apply_pca(n_components=3)
    clusterer.select_pca_components(acceptance_rate=95, show=False)
    assert 1 <= clusterer.pca_n_components <= 3
    assert (clusterer.pca_result.shape[1] == clusterer.pca_n_components)


def test_apply_kmeans():
    clusterer = create_clusterer()
    clusterer.apply_pca(n_components=2)
    clusterer.apply_kmeans(n_clusters=2, random_state=0)
    labels = clusterer.get_cluster_labels()
    assert len(labels) == 40
    assert len(np.unique(labels)) == 2
    assert ("cluster_labels" in clusterer.data.columns)


def test_cluster_centers():
    clusterer = create_clusterer()
    clusterer.apply_pca(n_components=2)
    clusterer.apply_kmeans(n_clusters=2, random_state=0)
    centers = (clusterer.get_cluster_centers())
    assert centers.shape == (2, 2)


def test_plot_elbow_curve(tmp_path, monkeypatch):
    monkeypatch.setattr(clustering_module, "figures_dir", tmp_path)
    clusterer = create_clusterer()
    clusterer.apply_pca(n_components=2)
    inertia = clusterer.plot_elbow_curve(max_k=4, random_state=0, show=False)
    assert len(inertia) == 4
    assert all(earlier >= later for earlier, later in zip(inertia, inertia[1:]))
    output_files = list(tmp_path.glob("elbow_plot_*"))
    assert len(output_files) == 1
