import numpy as np
import pandas as pd
import pytest

from msi_clustering.processing import DataProcessor


def test_create_dataframe(tmp_path):
    data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "100.0": [10.0, 20.0], "200.0": [30.0, 40.0], "300.0": [50.0, 60.0]})
    file_path = tmp_path / "sample.csv"
    data.to_csv(file_path, index=False)
    processor = DataProcessor(file_path)
    processor.create_dataframe()
    assert list(processor.mz_values) == ["100.0", "200.0", "300.0"]


def test_create_dataframe_missing_file(tmp_path):
    processor = DataProcessor(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        processor.create_dataframe()


def test_get_unique_coordinates():
    processor = DataProcessor("unused.csv")
    processor.data = pd.DataFrame({"X": [0, 0, 1, 1], "Y": [0, 1, 0, 1]})
    processor.get_unique_coordinates()
    np.testing.assert_array_equal(processor.x_unique, np.array([0, 1]))
    np.testing.assert_array_equal(processor.y_unique, np.array([0, 1]))


def test_clean_data():
    processor = DataProcessor("unused.csv")
    processor.data = pd.DataFrame({"X": [0, 1], "Y": [0, 1], "remove_me": [5, 6]})
    processor.clean_data(["remove_me"])
    assert "remove_me" not in processor.data.columns


def test_clean_data_invalid_column():
    processor = DataProcessor("unused.csv")
    processor.data = pd.DataFrame({"X": [0, 1], "Y": [0, 1]})
    with pytest.raises(KeyError):
        processor.clean_data(["missing_column"])


def test_normalize_by_tic():
    processor = DataProcessor("unused.csv")
    processor.data = pd.DataFrame(
        {"X": [0, 1], "Y": [0, 1], "100.0": [2.0, 3.0], "200.0": [2.0, 1.0]})
    processor.mz_values = ["100.0", "200.0"]
    processor.normalize_by_tic()
    np.testing.assert_allclose(processor.data["TIC"], [4.0, 4.0])
    np.testing.assert_allclose(processor.data["100.0"], [0.5, 0.75])
    np.testing.assert_allclose(processor.data["200.0"], [0.5, 0.25])
