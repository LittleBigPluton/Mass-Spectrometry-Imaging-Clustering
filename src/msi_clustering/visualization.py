###############################
#####   Import Libraries  #####
###############################
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from collections.abc import Mapping
from pathlib import Path
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .config import (
    figures_dir,
    figure_format,
    dpi_resolution,
    heatmap_size
)
# Import data processing library in order to build on top
from .processing import DataProcessor
####################################
##  Define Visualization Library  ##
####################################

class MSIVisualizer(DataProcessor):
    def __init__(self, file_path: str | Path | None = None, data: pd.DataFrame | None = None) -> None:
        super().__init__(file_path)
        self.data = data

    def save_plot(self,figure: Figure, value: str, plot_type: str) -> None:
        # Extract sample name from the data file's path
        if self.file_path is None:
            raise RuntimeError("File path is not defined.")
        sample_name = Path(self.file_path).stem
        save_path = figures_dir / f"{sample_name}_{value}_{plot_type}.{figure_format}"

        # Save the figure
        figure.savefig(save_path, format=figure_format, dpi=dpi_resolution)
        print(f"Plot saved as '{save_path}'.")

    def plot_heatmap(self, value: str, show: bool = False, save: bool = False) -> None:
        # To catch not defined value to plot
        data = self._require_data()
        if value not in data.columns:
            raise ValueError(f"{value!r} is not defined in the dataset.")
        # Create the pivot table to plot data as a heatmap
        pivot_table = data.pivot(index = "Y", columns = "X", values = value)
        # Create the intensity heatmap
        fig, ax = plt.subplots(figsize=heatmap_size)
        # Display the heatmap
        heatmap = ax.imshow(pivot_table,origin ='lower',cmap="CMRmap",interpolation='nearest')
        # Decide the label of the plot
        if value != "cluster_labels":
            # Create the colorbar and set its label
            fig.colorbar(heatmap, ax=ax, label='Intensity')
            # Set title and axis names
            ax.set_title(f'Heatmap of Molecule {value} Density')
        else:
            # Fetch unique labels and their associated colors from the colormap
            labels = sorted(data[value].unique())
            colors: list[tuple[float, float, float, float]] = []
            for label in labels:
                rgba = heatmap.cmap(heatmap.norm(label))
                colors.append((float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])))

            # Create a patch for each label
            patches = [mpatches.Patch(color=colors[i], label=f"Cluster {label}") for i, label in enumerate(labels)]
            ax.legend(handles=patches, title="Clusters", loc='best')
            ax.set_title("Cluster map of the data")
        ax.set_xlabel("X coordinates on the plane")
        ax.set_ylabel("Y coordinates on the plane")
        if show:
            plt.show()
        if save:
            self.save_plot(fig, value, "heatmap")
        plt.close(fig)

    def plot_cluster_comparison(self, labels_by_k: Mapping[int, NDArray[np.int_]], show: bool=False, save: bool=True) -> None:
        """Plot spatial cluster maps for several k values."""
        file_path = self._require_file_path()
        data = self._require_data()
        cluster_counts = sorted(labels_by_k.keys())
        fig, axes = plt.subplots(1, len(cluster_counts), figsize=(6 * len(cluster_counts), 6), constrained_layout=True)
        axes_array = np.atleast_1d(axes).ravel()

        for ax, n_clusters in zip(axes_array, cluster_counts):
            comparison_data = (data.copy())
            comparison_data["comparison_cluster_labels"] = labels_by_k[n_clusters]
            pivot_table = (comparison_data.pivot(index="Y", columns="X", values=("comparison_cluster_labels")))
            heatmap = ax.imshow(pivot_table, origin="lower", cmap="CMRmap", interpolation="nearest")
            labels = sorted(comparison_data["comparison_cluster_labels"].unique())
            colors = [heatmap.cmap(heatmap.norm(label)) for label in labels]
            patches = [mpatches.Patch(color=colors[index], label=f"Cluster {label}") for index, label in enumerate(labels)]
            ax.legend(handles=patches, title="Clusters", loc="best")
            ax.set_title(f"k = {n_clusters}")
            ax.set_xlabel("X coordinates on the plane")
            ax.set_ylabel("Y coordinates on the plane")

        fig.suptitle("Spatial K-means Cluster Comparison")
        if save:
            sample_name = Path(file_path).stem
            comparison_dir = (figures_dir / "comparison")
            comparison_dir.mkdir(parents=True, exist_ok=True)
            save_path = (comparison_dir / (f"{sample_name}_cluster_comparison.{figure_format}"))
            fig.savefig(save_path, format=figure_format, dpi=dpi_resolution, bbox_inches="tight")
            print(f"Comparison plot saved as '{save_path}'.")

        if show:
            plt.show()

        plt.close(fig)
