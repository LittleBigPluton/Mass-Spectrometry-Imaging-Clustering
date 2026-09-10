###############################
#####   Import Libraries  #####
###############################
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
# To save figures with given file's name
from pathlib import Path
from config import (
    figs_dir,
    figure_format,
    dpi_resolution,
    figure_size,
    heat_map_size,
)
# Import data processing library in order to build on top
from msi_clustering.processing import data_process
####################################
##  Define Visualization Library  ##
####################################

class visualize(data_process):
    def __init__(self, file_path=None, data=None):
        super().__init__(file_path)
        self.data = data

    def save_plot(self,figure, value, type):
        ####################################################################################################
        # Parameters:                                                                                     ##
        # - figure: The matplotlib figure to save.                                                        ##
        # - savepath: Name of the file to save the figure as.                                             ##
        # - directory: The directory where the figure should be saved. Defaults to the current directory. ##
        # - format: The file format (e.g., 'png', 'jpg', 'pdf', 'svg'). Defaults to 'png'.                ##
        # - dpi: The resolution in dots per inch. Defaults to 300 for high quality.                       ##
        ####################################################################################################
        # Extract sample name from the data file's path
        sample_name = Path(self.file_path).stem
        save_path = figs_dir / f"{sample_name}_{value}_{type}.{figure_format}"

        # Save the figure
        figure.savefig(save_path, format=figure_format, dpi=dpi_resolution)
        print(f"Plot saved as '{save_path}'.")

    def plot_heatmap(self, value, show = False, save = False):
        # To catch not defined value to plot
        if value not in self.data.columns:
            print(f"{value} is not defined in the data set.")
            raise(ValueError)
        # Create the pivot table to plot data as a heatmap
        pivot_table = self.data.pivot(index = "Y", columns = "X", values = value)
        # Create the intensity heatmap
        fig, ax = plt.subplots(figsize=heat_map_size)
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
            labels = sorted(self.data[value].unique())
            colors = [heatmap.cmap(heatmap.norm(label)) for label in labels]
            # Create a patch for each label
            patches = [mpatches.Patch(color=colors[i], label=f'Cluster {label}') for i, label in enumerate(labels)]
            ax.legend(handles=patches, title="Clusters", loc='best')
            ax.set_title("Cluster map of the data")
        ax.set_xlabel("X coordinates on the plane")
        ax.set_ylabel("Y coordinates on the plane")
        if show:
            plt.show()
        if save:
            self.save_plot(fig, value, "heatmap")

    def plot_cluster_comparison(self, labels_by_k, show=False, save=True):
        """Plot spatial cluster maps for several k values."""
        cluster_counts = sorted(labels_by_k.keys())
        fig, axes = plt.subplots(1, len(cluster_counts), figsize=(6 * len(cluster_counts), 6), constrained_layout=True)
        if len(cluster_counts) == 1:
            axes = [axes]

        for ax, n_clusters in zip(axes, cluster_counts):
            comparison_data = (self.data.copy())
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
            sample_name = Path(self.file_path).stem
            comparison_dir = (figs_dir / "comparison")
            comparison_dir.mkdir(parents=True, exist_ok=True)
            save_path = (comparison_dir / (f"{sample_name}_cluster_comparison.{figure_format}"))
            fig.savefig(save_path, format=figure_format, dpi=dpi_resolution, bbox_inches="tight")
            print(f"Comparison plot saved as '{save_path}'.")

        if show:
            plt.show()

        plt.close(fig)
