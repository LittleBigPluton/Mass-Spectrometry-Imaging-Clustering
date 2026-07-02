# Mass Spectrometry Imaging Clustering

DESI-MSI tissue region detection using PCA and K-means clustering.

This repository contains preprocessing mass spectrometry imaging data, reducing high-dimensional molecular intensity features with Principal Component Analysis (PCA), and detecting tissue-related spatial regions using K-means clustering. The project also includes visualization tools for plotting cluster maps and molecular intensity heatmaps from spatial MSI coordinates.

## Overview

Mass spectrometry imaging (MSI) produces spatially resolved molecular measurements. Each pixel or coordinate position contains many molecular intensity values, which makes the dataset high-dimensional and difficult to inspect directly.

This project applies an unsupervised learning workflow to:

1. Load DESI-MSI style tabular data.
2. Clean and reshape raw exported files into a structured format.
3. Apply PCA for dimensionality reduction.
4. Estimate a suitable number of K-means clusters using the elbow method.
5. Assign cluster labels to spatial coordinates.
6. Visualize the resulting tissue regions as heatmaps.

The main goal is to identify spatial patterns in DESI-MSI data without using manually annotated tissue labels.

## Table of Contents
1. [Overview](#project-overview)
2. [Main Features](#main-features)
3. [Repository Structure](#repository-structure)
4. [Workflow](#workflow)
5. [Expected Data Format](#expected-data-format)
6. [Installation](#installation)
7. [Optional System Dependency for Interactive Plots](#optional-system-dependency-for-interactive-plots)
8. [Configuration](#configuration)
9. [Usage](#usage)
10. [Data Preprocessing](#data-preprocessing)
11. [PCA Dimensionality Reduction](#pca-dimensionality-reduction)
12. [K-means Clustering](#k-means-clustering)
13. [Elbow Method](#elbow-method)
14. [Visualization](#visualization)
15. [Outputs](#outputs)
16. [Notes on Git Tracking](#notes-on-git-tracking)
17. [Troubleshooting](#troubleshooting)
18. [Future Improvements](#future-improvements)
19. [Technologies Used](#technologies-used)
20. [License](#license)
21. [Author](#author)
22. [References](#References)
    
## Main Features

* Preprocessing support for different MSI export formats.
* Data loading into a structured pandas DataFrame.
* PCA-based dimensionality reduction.
* Cumulative explained variance analysis.
* K-means clustering for tissue region detection.
* Elbow method for selecting cluster count.
* Cluster label extraction and cluster center inspection.
* Spatial heatmap visualization from X/Y coordinates.
* Output saving for processed data and generated figures.
* Package-style project organization with reusable modules.

## Repository Structure

```text
Mass-Spectrometry-Imaging-Clustering/
│
├── README.md
├── LICENSE
├── requirements.txt
├── config.py
├── run_clustering.py
│
├── data/
│   ├── raw/
│   │   └── example raw MSI data files
│   └── processed/
│       └── processed MSI data files
│
├── figs/
│   └── generated plots and heatmaps
│
└── msi_clustering/
    ├── __init__.py
    ├── processing.py
    ├── clustering.py
    └── visualization.py
```

### Main files

| File                | Purpose                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| `run_clustering.py` | Main script for running the full clustering workflow                        |
| `config.py`         | Central location for project paths and analysis parameters                  |
| `processing.py`     | Data loading, cleaning, preprocessing and normalization methods            |
| `clustering.py`     | PCA, explained variance analysis, K-means clustering and cluster utilities |
| `visualization.py`  | Heatmap and cluster visualization functions                                 |
| `requirements.txt`  | Python package dependencies                                                 |

## Workflow

The general workflow is:

```text
Raw MSI file
   ↓
Data cleaning / preprocessing
   ↓
Structured DataFrame with X, Y and molecular intensity columns
   ↓
PCA dimensionality reduction
   ↓
Explained variance inspection
   ↓
K-means clustering
   ↓
Cluster labels added to DataFrame
   ↓
Spatial heatmap / cluster map visualization
```

## Expected Data Format

After preprocessing, the data should have a tabular structure similar to:

```text
X     Y     123.456     255.233     ...
0.1   0.1   0.9         0.3         ...
0.1   0.2   0.3         0.4         ...
...   ...   ...         ...         ...
```

Where:

* `X` is the x-coordinate of the MSI pixel.
* `Y` is the y-coordinate of the MSI pixel.
* Each remaining column represents a molecular feature, often an `m/z` value.
* Each row represents one spatial measurement point.

The clustering pipeline expects spatial coordinates and numerical molecular intensity values.

## Installation

Clone the repository:

```bash
git clone https://github.com/LittleBigPluton/Mass-Spectrometry-Imaging-Clustering.git
cd Mass-Spectrometry-Imaging-Clustering
```

Create a virtual environment:

```bash
python3 -m venv venv_msi_clustering
```

Activate the environment:

```bash
source venv_msi_clustering/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Optional System Dependency for Interactive Plots

If `plt.show()` gives a warning such as:

```text
FigureCanvasAgg is non-interactive, and thus cannot be shown
```

install Tkinter on Linux:

```bash
sudo apt update
sudo apt install python3-tk
```

Test Tkinter with:

```bash
python -m tkinter
```

`tkinter` is usually not added to `requirements.txt` because it is installed as a system package, not through `pip`.

## Configuration

Project paths and main parameters can be defined in `config.py`.

Example:

```python
from pathlib import Path

project_root = Path(__file__).resolve().parent

data_dir = project_root / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"

figs_dir = project_root / "figures"

total_components = 96
explained_variance_threshold = 99
maximum_clusters = 10
```

Suggested parameter meanings:

```python
# Total number of PCA components to compute from the original feature space
total_components = 96

# Target cumulative explained variance percentage used to select retained PCA components
explained_variance_threshold = 95

# Maximum number of K-means clusters tested during the elbow method
maximum_clusters = 10
```

## Usage

Run the full clustering pipeline:

```bash
python3 run_clustering.py
```

A typical workflow inside `run_clustering.py` may look like:

```python
from config import total_components, explained_variance_threshold, maximum_clusters, raw_data_dir, processed_data_dir
from msi_clustering.clustering import cluster

file_path = raw_data_dir / "processed_Sample_PL.txt"

sample_data = cluster(file_path)

sample_data.create_data_frame()
sample_data.apply_PCA(n_components=total_components)
sample_data.get_PCA_features(explained_variance_threshold)
sample_data.find_optimal_clusters(max_k=maximum_clusters)
sample_data.apply_kmeans()
sample_data.plot_heatmap("cluster_labels", show=True, save=True)
```

## Data Preprocessing

Some MSI export files may require cleaning before clustering. The preprocessing step can be skipped if the data is already in the expected format:

```text
X | Y | molecular feature 1 | molecular feature 2 | ...
```

The project includes preprocessing logic for different example file structures, such as:

* Transposed MSI feature tables.
* Tab-separated MSI export files.
* Files requiring removal of metadata columns.
* Files requiring extraction of X/Y coordinates from index values.

Processed files should be saved under:

```text
data/processed/
```

Raw files should be kept under:

```text
data/raw/
```

This keeps the original data separate from cleaned analysis-ready data.

## PCA Dimensionality Reduction <sup>[1](https://en.wikipedia.org/wiki/Principal_component_analysis)</sup><sup>,[2](https://www.geeksforgeeks.org/machine-learning/reduce-data-dimentionality-using-pca-python/)</sup>

MSI datasets can contain a large number of molecular features. PCA is used to reduce the dimensionality of the dataset while preserving the main variance structure.

The PCA step computes a selected number of principal components:

```python
sample_data.apply_PCA(n_components=total_components)
```

The explained variance function helps determine how many components are needed to represent a desired percentage of total variance:

```python
sample_data.get_PCA_features(explained_variance_threshold)
```

For example, an explained variance threshold of `99` means that the analysis checks how many PCA components are needed to explain approximately 99% of the total variance. After PCA components plot showed up, 
a user input is saking on the terminal for desired total number of the features that will be used to analyze clustering. 15 components are sufficient to explain 99% of the total variance at the sample data 
so 15 (or more desired but not necessary) should be entered on the terminal. 

 ![Sample PCA plot](figures/pca_plot_20191017_liver_4v_75um_Analyte_1AFAMM_1_pixel_intensities.png)
 ![Sample PCA plot](https://github.com/LittleBigPluton/Mass-Spectrometry-Imaging-Clustering/blob/main/figures/pca_plot_Sample_PL.png)
 
## K-means Clustering <sup>[3](https://en.wikipedia.org/wiki/K-means_clustering)</sup><sup>,[4](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)</sup>

After PCA, K-means clustering is applied to the reduced feature space. The clustering step assigns each spatial data point to a cluster:

```python
sample_data.apply_kmeans()
```

The resulting cluster labels are added to the DataFrame as:

```text
cluster_labels
```

These labels can then be used to visualize spatial tissue regions.

## Elbow Method <sup>[5](https://en.wikipedia.org/wiki/Elbow_method_(clustering))</sup><sup>,[6](https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/)</sup>

The elbow method is used to inspect how K-means inertia changes with different numbers of clusters:

```python
sample_data.find_optimal_clusters(max_k=maximum_clusters)
```

The goal is to identify a reasonable number of clusters where the decrease in inertia starts to slow down.

![Elbow plot of sample data](figures/elbow_plot_20191017_liver_4v_75um_Analyte_1AFAMM_1_pixel_intensities.png)
![Elbow plot of sample data](figures/elbow_plot_Sample_PL.png)
## Visualization

The project supports heatmap-style visualization using spatial `X` and `Y` coordinates.

Example:

```python
sample_data.plot_heatmap("cluster_labels", show=True, save=True)
```

For molecular intensity visualization:

```python
sample_data.plot_heatmap("123.456", show=True, save=True)
```

For cluster maps:

* Each coordinate position is colored according to its assigned K-means cluster.
* The Y-axis orientation can be set so that spatial coordinates are displayed in a coordinate-plane-like layout.
* Generated figures can be saved automatically to the `figs/` directory.

![Sample cluster heatmap](figures/processed_20191017_liver_4v_75um_Analyte_1AFAMM_1_pixel_intensities_cluster_labels_heatmap.png)
![Sample cluster heatmap](figures/processed_Sample_PL_cluster_labels_heatmap.png)
## Outputs

Possible generated outputs include:

```text
figs/
├── Sample_PL_cluster_labels_heatmap.png
├── Sample_PL_pca_explained_variance.png
└── Sample_PL_elbow_method.png
```

Output names may vary depending on the input file and the plotting function.

## Notes on Git Tracking

Recommended files and folders to track:

```text
README.md
LICENSE
requirements.txt
config.py
run_clustering.py
msi_clustering/
data/raw/.gitkeep
data/processed/.gitkeep
figs/.gitkeep
```

Recommended files and folders to ignore:

```text
__pycache__/
*.pyc
venv*/
.env
data/raw/*
data/processed/*
figs/*.png
```

If sample data or output figures are intentionally included for demonstration, they can be committed. Otherwise, large raw data files and generated figures should usually be excluded from version control.

## Troubleshooting

### `FigureCanvasAgg is non-interactive`

This means Matplotlib is using a non-interactive backend. If you want plot windows to open, install Tkinter:

```bash
sudo apt install python3-tk
```

Or save plots directly using:

```python
plt.savefig("figure.png", dpi=300, bbox_inches="tight")
```

### `DataFrame.pivot() takes 1 positional argument but 4 were given`

Newer pandas versions require keyword arguments for `pivot()`:

```python
pivot_table = self.data.pivot(index="Y", columns="X", values=value)
```

### `DataFrame is highly fragmented`

This can happen when many columns are inserted one by one. A simple fix before adding a new column is:

```python
self.data = self.data.copy()
self.data["cluster_labels"] = self.cluster_labels
```

For larger workflows, it is better to collect new columns first and add them together with `pd.concat(axis=1)`.

## Future Improvements

Possible improvements for the project:

* Add command-line arguments for selecting input files and parameters.
* Add automated tests for preprocessing and clustering functions.
* Add example notebooks for exploratory analysis.
* Add support for additional MSI file formats.
* Add comparison with other clustering methods such as DBSCAN or Gaussian Mixture Models.
* Add quantitative validation metrics for clustering quality.
* Add documentation for each class and method.
* Add sample output figures to the README.
* Add a reproducible example dataset or small demo file.

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* PCA
* K-means clustering

## License

This project is licensed under the MIT License.

## Author

Created by [LittleBigPluton](https://github.com/LittleBigPluton).

## References
1. [Principal Component Analysis - Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
2. [Reduce Data Dimentionality by Using PCA - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/reduce-data-dimentionality-using-pca-python/)
3. [K-means Clustering - Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
4. [K-means Clustering Introduction - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)
5. [Elbow Method - Wikipedia](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
6. [Elbow Method for Optimal Value of k in K-means - GeekforGeeks](https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/)
