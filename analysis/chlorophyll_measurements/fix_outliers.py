# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Using data where multiple measurements are made and compare how each
measurement compares to the average and finds the 3-sigma outliers
and calculates and plots the R2 and histograms of the residues from
the average for each measurement
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import matplotlib.pyplot as plt

import pandas as pd


plt.style.use('bmh')

LEAVE = "Banana"
DATA_FOLDER = Path.cwd().parent.parent / "data" / "chlorophyll_data" / "collected_data"
print(DATA_FOLDER)


def get_data(leaf: str) -> pd.DataFrame:
    """ Load csv file with raw chlorophyll data for a set of leaves.

    Find a csv file with the name f"{leaf} Chlorophyll content.csv" in the DATA_FOLDER
    and get the "Leaf No." and "Total Chlorophyll (µg/cm2)"

    Args:
        leaf (str): name of the leave to use, for this project, "Banana", "Jasmine",
        "Mango", "Rice", or "Sugarcane"

    Returns:
        pd.DataFrame of the chlorophyll measurements

    """
    data_file = DATA_FOLDER / f"{leaf} Chlorophyll content.csv"
    _data = pd.read_csv(data_file, usecols=["Total Chlorophyll (µg/cm2)",
                                            "Leaf No."])
    print(type(_data))
    if "Leaf No." in _data.columns:
        _data = _data.ffill()
        _data["Leaf No."] = _data["Leaf No."].astype(int)
    return _data


if __name__ == '__main__':
    data = get_data(LEAVE)
    print(data)
