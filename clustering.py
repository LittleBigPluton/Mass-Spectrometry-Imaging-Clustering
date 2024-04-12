###################################
####      Import Libraries     ####
###################################
####      Data processing      ####
###################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
###################################

class data_process:

    def __init__(self, file_path):
        ########################################################################
        # Parameter:                                                          ##
        # - file_path: Data file's name or complete path to read and use data ##
        # - data: Pandas DataFrame to manipulate easily                       ##
        # - column_names: To extract column names from the file               ##
        # - Xunique: To create a meshgrid for colormesh, unique x values      ##
        # - Yunique: To create a meshgrid for colormesh, unique y values      ##
        # - Molecule: Desired m/z value to visualize                          ##
        ########################################################################

        #Initialize with file path and empty data attributes.
        self.file_path = file_path
        self.data = None
        self.column_names = None
        self.Xunique = None
        self.Yunique = None
        self.molecule = None

    def create_data_frame(self):
        # Read data from a tab-separated file and set up the DataFrame.
        # Change delimiter to use other seperations
        try:
            # Open file again to extract column names
            with open(self.file_path,'r') as file:
                lines = file.readlines()

            # Extract column names from the fourth line (index 3)
            # Split by tab and strip to remove any leading/trailing whitespace
            self.column_names = ["Index", "X", "Y"] + lines[3].strip().split('\t')

            # Use lines from the fifth line onwards (index 4) for data
            data_str = ''.join(lines[4:])

            # Convert the data string into a StringIO object
            # StringIO creates in-memory text stream from data_str
            # to give it to the DataFrame as a virtual file
            data_io = io.StringIO(data_str)

            # Read the data into a DataFrame
            self.data = pd.read_csv(data_io, delimiter='\t', header=None)

            # Rename the columns in the DataFrame with the extracted column names
            self.data.columns = self.column_names + list(self.data.columns[-2:])
            # Print out first 10 rows of the data to have a sight
            print("First the rows of the data is: ")
            print(self.data.head(10))
            print(f"Data includes {self.data.shape[0]} rows and {self.data.shape[1]} columns.")

        except FileNotFoundError as e:
            print("File not found. Please check the file path and try again.")
            print(e)
            exit()

    def get_column_names(self):
        print(self.data.columns)
        return self.data.columns

    def get_DataFrame(self):
        return self.data

    def set_molecule(self,molecule):
        self.molecule = str(molecule)

    def normalize_by_TIC(self):
        # Calculate Total Ion Current (TIC) for normalization,
        # excluding 'Index', 'X', 'Y', and the last two unnecessary columns.
        # self.column_names includes only Index, X, Y and rest of the columns
        # except last two not necessary columns
        self.data["TIC"] = self.data[self.column_names[3:]].sum(axis=1)
        # To normalize all data, uncomment the following code but computation time is longer
        self.data[self.column_names[3:]]=self.data[self.column_names[3:]].div(self.data["TIC"], axis=0)

####################################
######  Clustering Libraries  ######
####################################
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

####################################
####  Define Clustering Class   ####
####################################
class cluster(data_process):

    def apply_PCA(self, n_components=200):
        # Extract relevant data excluding 'Index', 'X', 'Y', and non-feature columns
        features = self.data[self.column_names[3:-2]]  # Adjust if extra columns were added during normalization
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(features)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")

    def get_PCA_features(self):
        """
        Plot the cumulative explained variance by the components to help decide on the number of components to retain.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Components')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        self.set_PCA_feature()
        self.apply_PCA(self.pca_n_components)

    def set_PCA_feature(self):
        while True:
            try:
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

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_k + 1), wcss)
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.grid(True)
        plt.show()
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
        # Apply K-means clustering on the PCA-reduced data.			    ##
	# Parameters:								    ##
        # - n_clusters: Optimal number of clusters determined from the elbow method.##
        ##############################################################################
        # Initialize KMeans with the optimal number of clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)

        # Fit KMeans on the PCA-reduced data
        self.cluster_labels = self.kmeans.fit_predict(self.pca_result)

        # Optionally, plot the clusters to visualize them if 2D PCA was applied
        if self.pca_result.shape[1] == 2:  # Check if PCA resulted in 2 dimensions
            plt.figure(figsize=(8, 6))
            plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], c=self.cluster_labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
            plt.title('Clusters in PCA-reduced Data')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()

    def get_cluster_labels(self):
        """
        Return the cluster labels assigned by K-means.

        Returns:
        - cluster_labels: Array of cluster labels.
        """
        return self.cluster_labels

    def get_cluster_centers(self):
        """
        Return the cluster centers in the PCA-reduced space.

        Returns:
        - cluster_centers: Coordinates of cluster centers.
        """
        return self.kmeans.cluster_centers_

    def plot_clusters(self):
        """
        Plots the clusters using the first two principal components.

        Parameters:
        - data: A 2D numpy array or DataFrame with the principal components.
        - labels: The cluster labels for each point in the data.
        """

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(self.data['X'],self.data['Y'], c=self.cluster_labels, cmap='viridis', alpha=0.5)
        plt.title('Cluster Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
