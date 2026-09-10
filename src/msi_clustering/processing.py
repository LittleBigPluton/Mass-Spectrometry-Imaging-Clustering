##################################
#####    Import Libraries    #####
##################################
import pandas as pd
import io
###################################
##  Define Data Process Library  ##
###################################


class DataProcessor:

    def __init__(self, file_path):
        ########################################################################
        # Parameter:                                                          ##
        # - file_path: Data file's name or complete path to read and use data ##
        # - data: Pandas DataFrame to manipulate easily                       ##
        # - column_names: To extract column names from the file               ##
        # - xunique: To create a meshgrid for colormesh, unique x values      ##
        # - yunique: To create a meshgrid for colormesh, unique y values      ##
        # - Molecule: Desired m/z value to visualize                          ##
        ########################################################################

        #Initialize with file path and empty data attributes.
        self.file_path = file_path
        self.data = None
        self.mz_values = None
        self.xunique = None
        self.yunique = None
        self.molecule = None

    def create_dataframe(self):
        try:
            # Read the cleaned data file from the given file path
            self.data = pd.read_csv(self.file_path)
            print(self.data.head())
            print(self.data.shape)
            # Extract mz values from the column names
            self.mz_values = self.data.columns[4:]

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Data file not found: {self.file_path}") from exc

    def get_unique_coordinates(self):
        # Get unique values of XY coordinates
        self.xunique = self.data['X'].unique()
        self.yunique = self.data['Y'].unique()

    def get_column_names(self):
        # Get whole column names of the file
        print(self.data.columns)
        return self.data.columns

    def get_dataframe(self):
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
        except KeyError as exc:
            raise KeyError(f"Could not remove columns {drop_columns}: {exc}") from exc

    def normalize_by_tic(self, all = True):
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

    def clean_transposed_msi_feature_table(self,processed_data_dir,sample_file_name):
        # Clean the data and get rid of unnecessary columns/rows and shape the data
        print(self.file_path)
        raw_data = pd.read_csv(self.file_path,skiprows=[0,1],index_col=None)
        raw_data = raw_data.drop(columns=['mol_formula','adduct','moleculeNames','moleculeIds'])
        raw_data = raw_data.T
        raw_data.columns = raw_data.iloc[0]
        raw_data = raw_data.drop(['mz'])
        # Extract mz values from the data
        mz_values = list(raw_data.columns)

        # Extract numbers and prepare for MultiIndex
        extracted_numbers = raw_data.index.str.extractall(r"(\d+)")[0].unstack()
        extracted_numbers.columns = ['X', 'Y']

        # Convert to integers
        extracted_numbers = extracted_numbers.astype(int)

        # Create a MultiIndex from the DataFrame columns
        multi_index = pd.MultiIndex.from_frame(extracted_numbers)

        # Assign the MultiIndex to your original DataFrame
        raw_data.index = multi_index
        data_file = processed_data_dir / f"processed_{sample_file_name}"
        raw_data.to_csv(data_file)
        raw_data = None
        # Change file_path from raw to processed
        self.file_path = data_file

    def clean_tab_separated_msi_export(self, processed_data_dir, sample_file_name):
        # Read data from a tab-separated file and set up the DataFrame.
        # Change delimiter to use other seperations
        try:
            # Open file again to extract column names
            with open(self.file_path,'r') as file:
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
            data_file = processed_data_dir / f"processed_{sample_file_name}"
            raw_data.to_csv(data_file, index=False)
            # Change file_path from raw to processed
            self.file_path = data_file


        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Data file not found: {self.file_path}") from exc
