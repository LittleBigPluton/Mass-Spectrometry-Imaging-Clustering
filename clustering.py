####################################
######  Clustering Libraries  ######
####################################
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from processing import data_process
import matplotlib.pyplot as plt
import numpy as np
####################################
####  Define Clustering Class   ####
####################################
class cluster(data_process):

    def apply_PCA(self, n_components=200):
        # Apply the PCA to get an idea how many features would be usefull to train unsupervised K-means ML algorithm
        # Extract relevant data excluding 'Index', 'X', 'Y'
        features = self.data[self.mz_values]
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(features)
        # Extract variance ratio with respect to n_components to have an insight about features
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")

    def get_PCA_features(self, acceptance_rate):
        # Plot the cumulative explained variance by the components to help decide on the number of components to retain.
        # Set figure size
        plt.figure(figsize=(10, 5))
        # Plot the explained variance ratio
        plt.plot(np.cumsum(self.explained_variance_ratio_))
        # Set axis and title
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Components')
        # Draw a line at the acceptance rate
        plt.axhline(y=acceptance_rate*0.01, color='r', linestyle='--', label=f'{acceptance_rate}% explained variance')
        # Set legend at the best location and draw grid lines
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        # Accoriding the graph, set the optimal n_components
        self.set_PCA_feature()
        # Apply PCA with optimal n_components
        self.apply_PCA(self.pca_n_components)

    def set_PCA_feature(self):
        while True:
            try:
                # Get n_components from the user
                self.pca_n_components = int(input("Please enter the PCA feature numbers: "))
                # Exit the loop if input is successfully converted to an integer
                break
            except ValueError:
                print("Entered invalid type for the PCA features. Please enter an integer.")


    def find_optimal_clusters(self, max_k=10):
        wcss = []  # Within-cluster sum of squares
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(self.pca_result)
            wcss.append(kmeans.inertia_)

        # Set figure size
        plt.figure(figsize=(10, 5))
        # Plot WCSS
        plt.plot(range(1, max_k + 1), wcss)
        # Set axis and title
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        # Draw grid lines
        plt.grid(True)
        plt.show()
        # Get number of clusters from the user
        self.get_cluster_numbers()
        plt.close()

    def get_cluster_numbers(self):
        while True:
            try:
                self.n_clusters = int(input("Please enter the cluster number: "))
                # Exit the loop if input is successfully converted to an integer
                break
            except ValueError:
                print("Entered invalid type for the cluster numbers. Please enter an integer.")


    def apply_kmeans(self):
    	##############################################################################
        # Apply K-means clustering on the PCA-reduced data.			                ##
	    # Parameters:								                                ##
        # - n_clusters: Optimal number of clusters determined from the elbow method.##
        ##############################################################################
        # Initialize KMeans with the optimal number of clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)

        # Fit KMeans on the PCA-reduced data
        self.cluster_labels = self.kmeans.fit_predict(self.pca_result)

        # Define a new column for claster labels
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
