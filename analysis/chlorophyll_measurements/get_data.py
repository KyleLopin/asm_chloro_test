# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import pandas as pd

# DATA_FOLDER = Path.cwd().parent.parent / "data" / "chlorophyll_data" / "collected_data"
DATA_FOLDER = Path(__file__).parent.parent.parent / "data" / "chlorophyll_data" / "collected_data"

def get_data(leaf: str, use_columns: tuple[str] = ("Total Chlorophyll (µg/cm2)",
                                                   "Spot", "Leaf No.")) -> pd.DataFrame:
    """ Load csv file with raw chlorophyll data for a set of leaves.

    Find a csv file with the name f"{leaf} Chlorophyll content.csv" in the DATA_FOLDER
    and get the "Leaf No." and "Total Chlorophyll (µg/cm2)".

    Args:
        leaf (str): Name of the leave to use, for this project, "Banana", "Jasmine",
        "Mango", "Rice", or "Sugarcane"
        use_columns (tuple[str]): Name of the columns to get from the data file to return.
        Use a tuple to prevent the defaults from mutating.

    Returns:
        pd.DataFrame: DataFrame of the file contents with the column names passed in.

    """
    # set the data file path
    data_file = DATA_FOLDER / f"{leaf} Chlorophyll content.csv"
    # read the data file, convert the use_columns to list if they are a tuple from defaults
    _data = pd.read_csv(data_file, usecols=list(use_columns))
    if "Leaf No." in _data.columns:
        _data = _data.ffill()
        _data["Leaf No."] = _data["Leaf No."].astype(int)
    return _data
