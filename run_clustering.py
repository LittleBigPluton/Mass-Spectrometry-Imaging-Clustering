#!/usr/bin/python3
# Running part of the clusterng script
#######################################
#####       Import Libraries      #####
#######################################
import pandas as pd
import io
from msi_clustering.visualization import visualize
from msi_clustering.clustering import cluster
from config import (
    total_components,
    explained_variance_threshold,
    maximum_clusters,
    n_clusters,
    random_state,
    raw_data_dir,
    processed_data_dir
)
from msi_clustering.processing import data_process
#######################################
### The following cleaning process can
### be skipped if the data is in the
### following form:
### X   |   Y | 123.456 | 255.233 | ...
### 0.1 | 0.1 | 0.9     | 0.3     | ...
### 0.1 | 0.2 | 0.3     | 0.4     | ...
### ... | ... | ...     | ...     | ...
########################################

########################################
#####     Data Preprocessing       #####
########################################
#####       Example File 1         #####
########################################
sample_data_name = "20191017_liver_4v_75um_Analyte_1AFAMM_1_pixel_intensities.csv"
msi_feature_data = raw_data_dir / sample_data_name
sample_data_1 = cluster(msi_feature_data)
sample_data_1.clean_transposed_msi_feature_table(processed_data_dir,sample_data_name)
# Create DataFrame
sample_data_1.create_data_frame()
# Normalize whole data by TIC
#sample_data_1.normalize_by_TIC()
# Apply PCA with a desired number of components
sample_data_1.apply_PCA(n_components = total_components)
# To visually determine the number of components to keep based on explained variance
sample_data_1.get_PCA_features(explained_variance_threshold, show=False)
print("Retained PCA components:", sample_data_1.pca_n_components)
# Find the optimal number of clusters using the elbow method
sample_data_1.find_optimal_clusters(max_k=maximum_clusters, random_state=random_state, show=False)
# Apply K-means
sample_data_1.apply_kmeans(n_clusters=n_clusters, random_state=random_state)
# To get cluster labels and centers
#print(sample_data_1.get_cluster_labels())
#print(sample_data_1.get_cluster_centers())
# In order to plot points as a scatter plot wrt cluster labels
#sample_data_1.plot_clusters()
# Call heatmap function form visualization library
visual_instance = visualize(file_path = sample_data_1.file_path, data = sample_data_1.data)
visual_instance.plot_heatmap("cluster_labels",show=True, save = True)

########################################
#####     Data Preprocessing       #####
########################################
#####       Example File 2         #####
########################################
tab_separated_data_name = "Sample_PL.txt"
tab_seperated_data = raw_data_dir / tab_separated_data_name
sample_data_2 = cluster(tab_seperated_data)
sample_data_2.clean_tab_separated_msi_export(processed_data_dir,tab_separated_data_name)
# Create DataFrame
sample_data_2.create_data_frame()
# Normalize whole data by TIC
#data.normalize_by_TIC()
# Apply PCA with a desired number of components
sample_data_2.apply_PCA(n_components = total_components)
# To visually determine the number of components to keep based on explained variance
sample_data_2.get_PCA_features(explained_variance_threshold, show=False)
print("Retained PCA components:", sample_data_2.pca_n_components)
# Find the optimal number of clusters using the elbow method
sample_data_2.find_optimal_clusters(max_k=maximum_clusters, random_state=random_state, show=False)
# Apply K-means
sample_data_2.apply_kmeans(n_clusters=n_clusters, random_state=random_state)
# To get cluster labels and centers
#print(sample_data_2.get_cluster_labels())
#print(sample_data_2.get_cluster_centers())
# In order to plot points as a scatter plot wrt cluster labels
#sample_data_2.plot_clusters()
# Call heatmap function form visualization library
visual_instance = visualize(file_path = sample_data_2.file_path, data = sample_data_2.data)
visual_instance.plot_heatmap("cluster_labels",show=True, save = True)
