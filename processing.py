##################################
#####    Import Libraries    #####
##################################

import pandas as pd
import io

###################################
##  Define Data Process Library  ##
###################################


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
        self.mz_values = None
        self.Xunique = None
        self.Yunique = None
        self.molecule = None

    def create_data_frame(self):
        try:
            # Read the cleaned data file from the given file path
            self.data = pd.read_csv(self.file_path)
            print(self.data.head())
            print(self.data.shape)
            # Extract mz values from the column names
            self.mz_values = self.data.columns[4:]

        except FileNotFoundError as e:
            print("File not found. Please check the file path and try again.")
            print(e)
            exit()

    def get_unique_coordinates(self):
        # Get unique values of XY coordinates
        self.Xunique = self.data['X'].unique()
        self.Yunique = self.data['Y'].unique()

    def get_column_names(self):
        # Get whole column names of the file
        print(self.data.columns)
        return self.data.columns

    def get_DataFrame(self):
        # TO get data as data frame
        return self.data

    def set_molecule(self,molecule):
        # To specify desired molecule to visualize
        self.molecule = str(molecule)

    def clean_data(self,drop_columns):
        # Drop columns are list include names of not desired columns
        # Make sure the list contains column names in the correct data type
        # like int, float, str and so on.
        try:
            self.data.drop(drop_columns, axis=1, inplace=True)
            print(f"Columns {drop_columns} were deleted from the DataFrame")
        except KeyError as e:
            print(f"Error removing columns: {e}")

    def normalize_by_TIC(self, all = True):
        # Calculate Total Ion Current (TIC) for normalization
        # Create a new column names as TIC and store whole total intensities by coordinates
        self.data["TIC"] = self.data[self.mz_values].sum(axis=1)
        if all:
            # To normalize all data, uncomment the following code but computation time is longer
            self.data[self.mz_values]=self.data[self.mz_values].div(self.data["TIC"], axis=0)
        else:
            # TO normalize just one molecule by TIC
            self.data[self.molecule] = self.data[self.molecule].div(self.data["TIC"], axis = 0)
        self.data = self.data.fillna(0)
        print(self.data.head())
