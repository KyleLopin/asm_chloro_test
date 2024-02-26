# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Module for loading raw chlorophyll data for a set of leaves.

This module provides functions to load CSV files containing raw chlorophyll data
for specific leaves. The function, get_data(), loads the data from the file,
selecting specified columns and returning them as a pandas DataFrame.

Standard Libraries:
    pathlib.Path: Provides object-oriented interface for working with filesystem paths.

Installed Libraries:
    pandas: A powerful data manipulation library.

Attributes:
    DATA_FOLDER (pathlib.Path): Path to the folder containing collected chlorophyll data.

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import pandas as pd

# DATA_FOLDER = Path.cwd().parent.parent / "data" / "chlorophyll_data" / "collected_data"
DATA_FOLDER = Path(__file__).parent.parent.parent / "data" / "chlorophyll_data" / "collected_data"


def get_data(leaf: str, use_columns: tuple[str] = ("Total Chlorophyll (µg/cm2)",
                                                   "Spot", "Leaf No."),
             file_ending: str = " Chlorophyll content.csv") -> pd.DataFrame:
    """ Load csv file with raw chlorophyll data for a set of leaves.

    Find a csv file with the name f"{leaf} Chlorophyll content.csv" in the DATA_FOLDER
    and get the "Leaf No." and "Total Chlorophyll (µg/cm2)".

    Args:
        leaf (str): Name of the leave to use, for this project, "Banana", "Jasmine",
        "Mango", "Rice", or "Sugarcane"
        use_columns (tuple[str]): Name of the columns to get from the data file to return.
        Use a tuple to prevent the defaults from mutating.
        file_ending (str): End of the file name to get the data from.
        Defaults to " Chlorophyll content.csv"

    Returns:
        pd.DataFrame: DataFrame of the file contents with the column names passed in.

    """
    # set the data file path
    data_file = DATA_FOLDER / f"{leaf}{file_ending}"
    # read the data file, convert the use_columns to list if they are a tuple from defaults
    _data = pd.read_csv(data_file, usecols=list(use_columns))
    if "Leaf No." in _data.columns:
        _data = _data.ffill()
        _data["Leaf No."] = _data["Leaf No."].astype(int)
    return _data
