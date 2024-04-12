#!/usr/bin/python3
# Main part of the clusterng script
import clustering

# Intialize file name
file_name = "20231120_QC_S22_PL_CLMC.txt"
# Load data set
data = clustering.cluster(file_name)
# Create DataFrame
data.create_data_frame()
# Normalize whole data by TIC
data.normalize_by_TIC()
# Apply PCA with a desired number of components
data.apply_PCA()
# To visually determine the number of components to keep based on explained variance
data.get_PCA_features()
# Find the optimal number of clusters using the elbow method
data.find_optimal_clusters(max_k=10)
# Apply K-means
data.apply_kmeans()
print(data.get_cluster_labels())
print(data.get_cluster_centers())
data.plot_clusters()
