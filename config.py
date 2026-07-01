from pathlib import Path

# Root project folder
project_root = Path(__file__).resolve().parent

# Data folders
data_dir = project_root / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"

# Cluster parameters
total_components = 96 # Total number of PCA components to compute from the original feature space
explained_variance_threshold = 99 # Target cumulative explained variance percentage used to select retained PCA components
maximum_clusters = 10 # Set maximum cluster number to plot elbow graph

# Figure folder
figs_dir = project_root / "figures"

# Figure parameters
figure_format = "png"
dpi_resolution = 300
figure_size = (10,5)
heat_map_size = (10,8)
