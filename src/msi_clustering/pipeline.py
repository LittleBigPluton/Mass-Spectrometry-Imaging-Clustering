#!/usr/bin/python3
# Running part of the clusterng script

#######################################
#####       Import Libraries      #####
#######################################
import pandas as pd
import io

from .visualization import visualize
from .clustering import cluster
from .processing import data_process
from .config import (
    total_components,
    explained_variance_threshold,
    maximum_clusters,
    n_clusters,
    random_state,
    raw_data_dir,
    processed_data_dir,
    minimum_clusters,
    silhouette_sample_size,
    stability_random_states,
    reports_dir,
    comparison_clusters
)
from .evaluation import (
    evaluate_kmeans_candidates,
    evaluate_kmeans_stability,
    generate_candidate_cluster_labels
)


# Make sure reports directory exist or create
reports_dir.mkdir(parents=True, exist_ok=True)

#######################################
### The following cleaning process can
### be skipped if the data is in the
### following form:
### X   |   Y | 123.456 | 255.233 | ...
### 0.1 | 0.1 | 0.9     | 0.3     | ...
### 0.1 | 0.2 | 0.3     | 0.4     | ...
### ... | ... | ...     | ...     | ...
########################################
def main():
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
    # Apply spatial cluster comparison
    comparison_labels_1 = (generate_candidate_cluster_labels(sample_data_1.pca_result, cluster_counts=(comparison_clusters), random_state=random_state))
    # To visualize spatial cluster comparison
    visual_instance_1 = visualize(file_path=sample_data_1.file_path, data=sample_data_1.data)
    visual_instance_1.plot_cluster_comparison(comparison_labels_1, show=False, save=True)
    # Find the optimal number of clusters using the elbow method
    sample_data_1.find_optimal_clusters(max_k=maximum_clusters, random_state=random_state, show=False)
    # Evaluate candidate K-means cluster counts using clustering quality metrics
    evaluation_1 = evaluate_kmeans_candidates(sample_data_1.pca_result, min_clusters=minimum_clusters, max_clusters=maximum_clusters,
                                              random_state=random_state, silhouette_sample_size=silhouette_sample_size)
    # Display clustering evaluation results
    print("\nSample 1 clustering evaluation:")
    print(evaluation_1.to_string(index=False))
    # Save clustering evaluation metrics to a CSV report
    evaluation_1.to_csv(reports_dir / "sample_1_clustering_metrics.csv", index=False)
    # Evaluate K-means clustering stability across different random initializations
    stability_1 = evaluate_kmeans_stability(sample_data_1.pca_result, n_clusters=n_clusters, random_states=stability_random_states)
    # Display clustering stability results for the selected number of clusters
    print(f"\nSample 1 \nk={n_clusters} stability:")
    print(stability_1)
    # Apply K-means
    sample_data_1.apply_kmeans(n_clusters=n_clusters, random_state=random_state)
    # To get cluster labels and centers
    #print(sample_data_1.get_cluster_labels())
    #print(sample_data_1.get_cluster_centers())
    # In order to plot points as a scatter plot wrt cluster labels
    #sample_data_1.plot_clusters()
    # Call heatmap function form visualization library
    visual_instance_1 = visualize(file_path = sample_data_1.file_path, data = sample_data_1.data)
    visual_instance_1.plot_heatmap("cluster_labels",show=True, save = True)

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
    # Apply spatial cluster comparison
    comparison_labels_2 = (generate_candidate_cluster_labels(sample_data_2.pca_result, cluster_counts=(comparison_clusters), random_state=random_state))
    # To visualize spatial cluster comparison
    visual_instance_2 = visualize(file_path=sample_data_2.file_path, data=sample_data_2.data)
    visual_instance_2.plot_cluster_comparison(comparison_labels_2, show=False, save=True)
    # Find the optimal number of clusters using the elbow method
    sample_data_2.find_optimal_clusters(max_k=maximum_clusters, random_state=random_state, show=False)
    # Evaluate candidate K-means cluster counts using clustering quality metrics
    evaluation_2 = evaluate_kmeans_candidates(sample_data_2.pca_result, min_clusters=minimum_clusters, max_clusters=maximum_clusters,
                                              random_state=random_state, silhouette_sample_size=silhouette_sample_size)
    # Display clustering evaluation results
    print("\nSample 2 clustering evaluation:")
    print(evaluation_2.to_string(index=False))
    # Save clustering evaluation metrics to a CSV report
    evaluation_2.to_csv(reports_dir / "sample_2_clustering_metrics.csv", index=False)
    # Evaluate K-means clustering stability across different random initializations
    stability_2 = evaluate_kmeans_stability(sample_data_2.pca_result, n_clusters=n_clusters, random_states=stability_random_states)
    # Display clustering evaluation results
    print(f"\nSample 2 \nk={n_clusters} stability:")
    print(stability_2)
    # Apply K-means
    sample_data_2.apply_kmeans(n_clusters=n_clusters, random_state=random_state)
    # To get cluster labels and centers
    #print(sample_data_2.get_cluster_labels())
    #print(sample_data_2.get_cluster_centers())
    # In order to plot points as a scatter plot wrt cluster labels
    #sample_data_2.plot_clusters()
    # Call heatmap function form visualization library
    visual_instance_2 = visualize(file_path = sample_data_2.file_path, data = sample_data_2.data)
    visual_instance_2.plot_heatmap("cluster_labels",show=True, save = True)

if __name__ == "__main__":
    main()
