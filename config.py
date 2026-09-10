from pathlib import Path

# Root project folder
project_root = Path(__file__).resolve().parent

# Data folders
data_dir = project_root / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"

# PCA configuration
total_components = 96
explained_variance_threshold = 99

# K-means configuration
n_clusters = 4
maximum_clusters = 10
random_state = 0

# Figure folder
figs_dir = project_root / "figures"

# Figure parameters
figure_format = "png"
dpi_resolution = 300
figure_size = (10,5)
heat_map_size = (10,8)
