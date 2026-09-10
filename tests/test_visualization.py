import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import msi_clustering.visualization as visualization_module
from msi_clustering.visualization import MSIVisualizer


def create_spatial_data():
    return pd.DataFrame({"X": [0, 1, 2,
                               0, 1, 2,
                               0, 1, 2],
                         "Y": [0, 0, 0,
                               1, 1, 1,
                               2, 2, 2],
                         "100.0": [1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0,
                                   7.0, 8.0, 9.0],
                         "cluster_labels": [0, 0, 1,
                                            0, 1, 1,
                                            2, 2, 2]})


def test_plot_heatmap(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization_module, "figures_dir", tmp_path)
    visualizer = MSIVisualizer(file_path=tmp_path / "sample.csv", data=create_spatial_data())
    visualizer.plot_heatmap("cluster_labels", show=False, save=True)
    output_files = list(tmp_path.glob("*cluster_labels_heatmap*"))
    assert len(output_files) == 1


def test_plot_heatmap_invalid_column(tmp_path):
    visualizer = MSIVisualizer(file_path=tmp_path / "sample.csv", data=create_spatial_data())
    with pytest.raises(ValueError, match="not defined"):
        visualizer.plot_heatmap("missing_column", show=False, save=False)


def test_plot_cluster_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization_module, "figures_dir", tmp_path)
    data = create_spatial_data()
    visualizer = MSIVisualizer(file_path=tmp_path / "sample.csv", data=data)
    labels_by_k = {2: np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]), 3: np.array([0, 0, 1, 0, 1, 1, 2, 2, 2]), 4: np.array([0, 0, 1, 0, 1, 1, 2, 3, 3])}
    visualizer.plot_cluster_comparison(labels_by_k, show=False, save=True)
    output_files = list((tmp_path / "comparison").glob("*cluster_comparison*"))
    assert len(output_files) == 1
