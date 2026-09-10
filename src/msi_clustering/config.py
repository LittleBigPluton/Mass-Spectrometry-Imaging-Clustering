from pathlib import Path

# Root project folder
project_root = Path(__file__).resolve().parents[2]

# Data folders
raw_data_dir = project_root / "data" / "raw"
processed_data_dir = project_root / "data" / "processed"
figs_dir = project_root / "figures"
reports_dir = project_root / "reports"

# PCA configuration
total_components = 96
explained_variance_threshold = 99

# K-means configuration
n_clusters = 4
maximum_clusters = 10
random_state = 0

# Clustering evaluation
minimum_clusters = 2
silhouette_sample_size = 5000
stability_random_states = (0, 1, 2, 3, 4)

# Spatial cluster comparison
comparison_clusters = (2, 3, 4)

# Figure parameters
figure_format = "png"
dpi_resolution = 300
figure_size = (10,5)
heat_map_size = (10,8)
