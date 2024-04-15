#!/usr/bin/python3
# Main part of the clusterng script
#######################################
#####       Import Libraries      #####
#######################################
import pandas as pd
from visualization import visualize
from clustering import cluster
import io
#######################################
### The following cleaning process can
### be skipped if the data is in the
##  following form:
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

# Clean the data and get rid of unnecessary columns/rows and shape the data

file_path = "20191017_liver_4v_75um_Analyte_1AFAMM_1_pixel_intensities.csv"
raw_data = pd.read_csv(file_path,skiprows=[0,1],index_col=None)
raw_data = raw_data.drop(columns=['mol_formula','adduct','moleculeNames','moleculeIds'])
raw_data = raw_data.T
raw_data.columns = raw_data.iloc[0]
raw_data = raw_data.drop(['mz'])
# Extract mz values from the data
mz_values = list(raw_data.columns)

# Extract numbers and prepare for MultiIndex
extracted_numbers = raw_data.index.str.extractall('(\d+)')[0].unstack()
extracted_numbers.columns = ['X', 'Y']

# Convert to integers
extracted_numbers = extracted_numbers.astype(int)

# Create a MultiIndex from the DataFrame columns
multi_index = pd.MultiIndex.from_frame(extracted_numbers)

# Assign the MultiIndex to your original DataFrame
raw_data.index = multi_index
file_path = "cleaned_"+file_path
raw_data.to_csv(file_path)
raw_data = None

########################################
#####     Data Preprocessing       #####
########################################
#####       Example File 2         #####
########################################

file_path = "Sample_PL.txt"
# Read data from a tab-separated file and set up the DataFrame.
# Change delimiter to use other seperations
try:
    # Open file again to extract column names
    with open(file_path,'r') as file:
        lines = file.readlines()

    # Extract column names from the fourth line (index 3)
    # Split by tab and strip to remove any leading/trailing whitespace
    column_names = ["Index", "X", "Y"] + lines[3].strip().split('\t')

    # Use lines from the fifth line onwards (index 4) for data
    data_str = ''.join(lines[4:])

    # Convert the data string into a StringIO object
    # StringIO creates in-memory text stream from data_str
    # to give it to the DataFrame as a virtual file
    data_io = io.StringIO(data_str)

    # Read the data into a DataFrame
    raw_data = pd.read_csv(data_io, delimiter='\t', header=None)

    # Rename the columns in the DataFrame with the extracted column names
    raw_data.columns = column_names + list(raw_data.columns[-2:])
    # Print out first 10 rows of the data to have a sight
    print("First the rows of the data is: ")
    print(raw_data.head(10))
    print(f"Data includes {raw_data.shape[0]} rows and {raw_data.shape[1]} columns.")
    file_path = "cleaned_"+file_path
    raw_data.to_csv(file_path)


except FileNotFoundError as e:
    print("File not found. Please check the file path and try again.")
    print(e)
    exit()

# Intialize file name
file_name = "cleaned_Sample_PL.txt"
# Load data set
data = cluster(file_name)
# Create DataFrame
data.create_data_frame()
# Normalize whole data by TIC
#data.normalize_by_TIC()
# Apply PCA with a desired number of components
data.apply_PCA(n_components = 96)
# To visually determine the number of components to keep based on explained variance
data.get_PCA_features(95)
# Find the optimal number of clusters using the elbow method
data.find_optimal_clusters(max_k=10)
# Apply K-means
data.apply_kmeans()
# To get cluster labels and centers
#print(data.get_cluster_labels())
#print(data.get_cluster_centers())
# In order to plot points as a scatter plot wrt cluster labels
#data.plot_clusters()
# Call heatmap function form visualization library
visual_instance = visualize(file_name)
visual_instance.data = data.data
visual_instance.plot_heatmap("cluster_labels",show=True, save = True)
