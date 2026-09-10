####################################
######  Clustering Libraries  ######
####################################
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .processing import DataProcessor
from .config import (
    figures_dir,
    figure_format,
    dpi_resolution,
    figure_size
)


####################################
####  Define Clustering Class   ####
####################################
class MSIClusterer(DataProcessor):
    pca_result: NDArray[np.float64]
    explained_variance_ratio_: NDArray[np.float64]
    pca_n_components: int
    n_clusters: int
    kmeans: KMeans
    cluster_labels: NDArray[np.int_]

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path)
        self.sample_name = Path(file_path).stem

    def apply_pca(self, n_components: int=200) -> None:
        # Apply the PCA to get an idea how many features would be usefull to train unsupervised K-means ML algorithm
        # Extract relevant data excluding 'Index', 'X', 'Y'
        data = self._require_data()
        if self.mz_values is None:
            raise RuntimeError("m/z columns have not been initialized.")
        features = data[self.mz_values]
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(features)
        # Extract variance ratio with respect to n_components to have an insight about features
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")

    def select_pca_components(self, acceptance_rate: float, show: bool = False) -> None:
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        threshold = acceptance_rate / 100.0
        self.pca_n_components = (int(np.searchsorted(cumulative_variance, threshold)) + 1)

        plt.figure(figsize=figure_size)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
        plt.axhline(y=threshold, color="r", linestyle="--", label=f"{acceptance_rate}% explained variance")
        plt.axvline(x=self.pca_n_components, color="k", linestyle=":", label=(f"{self.pca_n_components} components"))
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Explained Variance by PCA Components")
        plt.legend(loc="best")
        plt.grid(True)

        save_path = (figures_dir / (f"pca_plot_{self.sample_name}.{figure_format}"))
        plt.savefig(save_path, format=figure_format, dpi=dpi_resolution, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        self.apply_pca(n_components=self.pca_n_components)


    def plot_elbow_curve(self, max_k: int = 10, random_state: int = 0, show: bool = False) -> list[float]:
        wcss: list[float] = []
        cluster_range = range(1, max_k + 1)
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=random_state)
            kmeans.fit(self.pca_result)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=figure_size)
        plt.plot(cluster_range, wcss, marker="o")
        plt.title("Elbow Method for K-means")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Within-Cluster Sum of Squares")
        plt.grid(True)

        save_path = (figures_dir / (f"elbow_plot_{self.sample_name}.{figure_format}"))
        plt.savefig(save_path, format=figure_format, dpi=dpi_resolution, bbox_inches="tight")
        if show:
            plt.show()

        plt.close()
        return wcss


    def apply_kmeans(self, n_clusters: int, random_state: int = 0) -> None:
        ##############################################################################
        # Apply K-means clustering on the PCA-reduced data.			                ##
    	# Parameters:								                                ##
        # - n_clusters: Optimal number of clusters determined from the elbow method.##
        ##############################################################################
        # Initialize number of the clusters
        self.n_clusters = n_clusters

        # Initialize KMeans with the optimal number of clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=random_state)

        # Fit KMeans on the PCA-reduced data
        self.cluster_labels = self.kmeans.fit_predict(self.pca_result)

        # Define a new column for cluster labels
        data = self._require_data()
        self.data = data.copy()
        self.data['cluster_labels'] = self.cluster_labels

    def get_cluster_labels(self) -> NDArray[np.int_]:
        # Return the cluster labels assigned by K-means
        return self.cluster_labels

    def get_cluster_centers(self) -> NDArray[np.int_]:
        # Return the cluster centers in the PCA-reduced space
        # Coordinates of cluster centers
        return self.kmeans.cluster_centers_

    def plot_clusters(self) -> None:
        # set figure size
        plt.figure(figsize=(8,6))
        # Reverse the Y column to have better visualization
        data = self._require_data()
        data['Y'] = data['Y'].iloc[::-1].reset_index(drop=True)
        # Plot scatter points
        plt.scatter(data['X'],data['Y'], c=self.cluster_labels, cmap='viridis', alpha=0.5)
        # Set axis and title
        plt.title('Cluster Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
