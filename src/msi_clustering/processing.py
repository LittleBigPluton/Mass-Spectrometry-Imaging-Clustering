##################################
#####    Import Libraries    #####
##################################
from pathlib import Path
from typing import Any

import io
import pandas as pd

from numpy.typing import NDArray
###################################
##  Define Data Process Library  ##
###################################


class DataProcessor:

    def __init__(self, file_path: str | Path | None) -> None:
        ########################################################################
        # Parameter:                                                          ##
        # - file_path: Data file's name or complete path to read and use data ##
        # - data: Pandas DataFrame to manipulate easily                       ##
        # - column_names: To extract column names from the file               ##
        # - x_unique: To create a meshgrid for colormesh, unique x values      ##
        # - y_unique: To create a meshgrid for colormesh, unique y values      ##
        # - Molecule: Desired m/z value to visualize                          ##
        ########################################################################

        #Initialize with file path and empty data attributes.
        self.file_path: str | Path | None = file_path
        self.data: pd.DataFrame | None = None
        self.mz_values: pd.Index | None = None
        self.x_unique: NDArray[Any] | None = None
        self.y_unique: NDArray[Any] | None = None
        self.molecule: str | None = None

    def _require_data(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("Data has not been loaded.")
        return self.data


    def _require_file_path(self) -> str | Path:
        if self.file_path is None:
            raise RuntimeError("File path is not defined.")
        return self.file_path

    def create_dataframe(self) -> None:
        try:
            # Read the cleaned data file from the given file path
            file_path = self._require_file_path()
            self.data = pd.read_csv(file_path)
            print(self.data.head())
            print(self.data.shape)
            # Extract mz values from the column names
            self.mz_values = self.data.columns[2:]

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Data file not found: {self.file_path}") from exc

    def get_unique_coordinates(self) ->  None:
        # Get unique values of XY coordinates
        data = self._require_data()
        self.x_unique = data["X"].unique()
        self.y_unique = data["Y"].unique()

    def get_column_names(self) -> pd.Index:
        # Get whole column names of the file
        data = self._require_data()
        print(data.columns)
        return data.columns

    def get_dataframe(self) -> pd.DataFrame:
        return self._require_data()

    def set_molecule(self, molecule: object) -> None:
        self.molecule = str(molecule)

    def clean_data(self, drop_columns: list[str]) -> None:
        # Drop columns are list include names of not desired columns
        # Make sure the list contains column names in the correct data type
        # like int, float, str and so on.
        data = self._require_data()
        try:
            self.data = data.drop(columns=drop_columns)
            print(f"Columns {drop_columns} were deleted from the DataFrame")
        except KeyError as exc:
            raise KeyError(f"Could not remove columns {drop_columns}: {exc}") from exc

    def normalize_by_tic(self, all: bool = True) -> None:
        data = self._require_data()

        if self.mz_values is None:
            raise RuntimeError("m/z columns have not been initialized.")

        # Calculate Total Ion Current (TIC) for normalization
        data["TIC"] = data[self.mz_values].sum(axis=1)

        if all:
            # Normalize all m/z values by TIC
            data[self.mz_values] = data[self.mz_values].div(data["TIC"], axis=0)
        else:
            if self.molecule is None:
                raise RuntimeError("No molecule has been selected.")

            # Normalize only the selected molecule by TIC
            data[self.molecule] = data[self.molecule].div(data["TIC"], axis=0)

        self.data = data.fillna(0)
        print(self.data.head())

    def clean_transposed_msi_feature_table(self, processed_data_dir: Path, sample_file_name: str) -> None:
        # Clean the data and get rid of unnecessary columns/rows and shape the data
        file_path = self._require_file_path()
        raw_data = pd.read_csv(file_path,skiprows=[0,1],index_col=None)
        raw_data = raw_data.drop(columns=['mol_formula','adduct','moleculeNames','moleculeIds'])
        raw_data = raw_data.T
        raw_data.columns = raw_data.iloc[0]
        raw_data = raw_data.drop(['mz'])

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
        # Change file_path from raw to processed
        self.file_path = data_file

    def clean_tab_separated_msi_export(self, processed_data_dir: Path, sample_file_name: str) -> None:
        # Read data from a tab-separated file and set up the DataFrame.
        # Change delimiter to use other seperations
        file_path = self._require_file_path()
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
            data_file = processed_data_dir / f"processed_{sample_file_name}"
            raw_data = raw_data.drop(['Index',*list(raw_data.columns[-2:])], axis=1)
            raw_data.to_csv(data_file, index=False)
            # Change file_path from raw to processed
            self.file_path = data_file


        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Data file not found: {self.file_path}") from exc
