###################################
####      Import Libraries     ####
###################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



####################################
####  Define Clustering Class   ####
####################################
 class data_process:

    def __init__(self, file_path):
        ########################################################################
        # Parameter:                                                          ##
        # - file_path: Data file's name or complete path to read and use data ##
        # - data: Pandas DataFrame to manipulate easily                       ##
        # - column_names: To extract column names from the file               ##
        # - Xunique: To create a meshgrid for colormesh, unique x values      ##
        # - Yunique: To create a meshgrid for colormesh, unique y values      ##
        # - Molecule: Desired m/z value to visualize                          ##
        ########################################################################

        #Initialize with file path and empty data attributes.
        self.file_path = file_path
        self.data = None
        self.column_names = None
        self.Xunique = None
        self.Yunique = None
        self.molecule = None

    def create_data_frame(self):
        # Read data from a tab-separated file and set up the DataFrame.
        # Change delimiter to use other seperations
        try:
            # Open file again to extract column names
            with open(self.file_path,'r') as file:
                lines = file.readlines()

            # Extract column names from the fourth line (index 3)
            # Split by tab and strip to remove any leading/trailing whitespace
            self.column_names = ["Index", "X", "Y"] + lines[3].strip().split('\t')

            # Use lines from the fifth line onwards (index 4) for data
            data_str = ''.join(lines[4:])

            # Convert the data string into a StringIO object
            # StringIO creates in-memory text stream from data_str
            # to give it to the DataFrame as a virtual file
            data_io = io.StringIO(data_str)

            # Read the data into a DataFrame
            self.data = pd.read_csv(data_io, delimiter='\t', header=None)

            # Rename the columns in the DataFrame with the extracted column names
            self.data.columns = self.column_names + list(self.data.columns[-2:])
            # Print out first 10 rows of the data to have a sight
            print("First the rows of the data is: ")
            print(self.data.head(10))
            print(f"Data includes {self.data.shape[0]} rows and {self.data.shape[1]} columns.")

        except FileNotFoundError as e:
            print("File not found. Please check the file path and try again.")
            print(e)
            exit()

    def get_column_names(self):
        print(self.data.columns)
        return self.data.columns

    def get_DataFrame(self):
        return self.data

    def set_molecule(self,molecule):
        self.molecule = str(molecule)

    def normalize_by_TIC(self):
        # Calculate Total Ion Current (TIC) for normalization,
        # excluding 'Index', 'X', 'Y', and the last two unnecessary columns.
        # self.column_names includes only Index, X, Y and rest of the columns
        # except last two not necessary columns
        self.data["TIC"] = self.data[self.column_names[3:]].sum(axis=1)
        # To normalize all data, uncomment the following code but computation time is longer
        # self.data[self.column_names[3:]]=self.data[self.column_names[3:]].div(self.data["TIC"], axis=0)
        self.data[self.molecule] = self.data[self.molecule].div(self.data["TIC"], axis = 0)

class cluster(data_process):
    def asd:
        return 
