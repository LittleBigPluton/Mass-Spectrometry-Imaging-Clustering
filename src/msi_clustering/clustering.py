####################################
######  Clustering Libraries  ######
####################################
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .processing import data_process
from .config import (
    figs_dir,
    figure_format,
    dpi_resolution,
    figure_size
)


####################################
####  Define Clustering Class   ####
####################################
class cluster(data_process):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.sample_name = Path(self.file_path).stem

    def apply_PCA(self, n_components=200):
        # Apply the PCA to get an idea how many features would be usefull to train unsupervised K-means ML algorithm
        # Extract relevant data excluding 'Index', 'X', 'Y'
        features = self.data[self.mz_values]
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(features)
        # Extract variance ratio with respect to n_components to have an insight about features
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")

    def get_PCA_features(self, acceptance_rate, show=False):
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

        save_path = (figs_dir / (f"pca_plot_{self.sample_name}.{figure_format}"))
        plt.savefig(save_path, format=figure_format, dpi=dpi_resolution, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        self.apply_PCA(n_components=self.pca_n_components)

    def set_PCA_feature(self):
        while True:
            try:
                # Get n_components from the user
                self.pca_n_components = int(input("Please enter the PCA feature numbers: "))
                # Exit the loop if input is successfully converted to an integer
                break
            except ValueError:
                print("Entered invalid type for the PCA features. Please enter an integer.")


    def find_optimal_clusters(self, max_k=10, random_state=0, show=False):
        wcss = []
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

        save_path = (figs_dir / (f"elbow_plot_{self.sample_name}.{figure_format}"))
        plt.savefig(save_path, format=figure_format, dpi=dpi_resolution, bbox_inches="tight")
        if show:
            plt.show()

        plt.close()
        return wcss


    def apply_kmeans(self, n_clusters, random_state=0):
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
        self.data = self.data.copy()
        self.data['cluster_labels'] = self.cluster_labels

    def get_cluster_labels(self):
        # Return the cluster labels assigned by K-means
        return self.cluster_labels

    def get_cluster_centers(self):
        # Return the cluster centers in the PCA-reduced space
        # Coordinates of cluster centers
        return self.kmeans.cluster_centers_

    def plot_clusters(self):
        # set figure size
        plt.figure(figsize=(8,6))
        # Reverse the Y column to have better visualization
        self.data['Y'] = self.data['Y'].iloc[::-1].reset_index(drop=True)
        # Plot scatter points
        scatter = plt.scatter(self.data['X'],self.data['Y'], c=self.cluster_labels, cmap='viridis', alpha=0.5)
        # Set axis and title
        plt.title('Cluster Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
