###############################
#####   Import Libraries  #####
###############################
import matplotlib.pyplot as plt
import numpy as np
# Import data processing library in order to build on top
from processing import data_process
import matplotlib.patches as mpatches

####################################
##  Define Visualization Library  ##
####################################

class visualize(data_process):
    def save_plot(self,figure, value, type, directory = '.',format = 'png', dpi = 300):
        ####################################################################################################
        # Parameters:                                                                                     ##
        # - figure: The matplotlib figure to save.                                                        ##
        # - savepath: Name of the file to save the figure as.                                             ##
        # - directory: The directory where the figure should be saved. Defaults to the current directory. ##
        # - format: The file format (e.g., 'png', 'jpg', 'pdf', 'svg'). Defaults to 'png'.                ##
        # - dpi: The resolution in dots per inch. Defaults to 300 for high quality.                       ##
        ####################################################################################################
        # Extract sample name from the data file's path
        sample_name = self.file_path[:-4]
        save_path = str(f"{sample_name}_{value}_{type}.{format}")

        # Save the figure
        figure.savefig(save_path, format=format, dpi=dpi)
        print(f"Plot saved as '{save_path}'.")

    def plot_heatmap(self, value, figure_size = (10,8), show = False, save = False):
        # To catch not defined value to plot
        if value not in self.data.columns:
            print(f"{value} is not defined in the data set.")
            raise(ValueError)
        # Create the pivot table to plot data as a heatmap
        pivot_table = self.data.pivot("Y", "X",value)
        # Create the intensity heatmap
        fig, ax = plt.subplots(figsize=figure_size)
        # Display the heatmap
        heatmap = ax.imshow(pivot_table,origin ='upper',cmap="CMRmap",interpolation='nearest')
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
