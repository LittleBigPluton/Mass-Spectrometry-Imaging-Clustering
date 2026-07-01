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
