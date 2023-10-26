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
        pd.DataFrame: of the chlorophyll measurements

    """
    data_file = DATA_FOLDER / f"{leaf} Chlorophyll content.csv"
    _data = pd.read_csv(data_file, usecols=["Total Chlorophyll (µg/cm2)",
                                            "Leaf No."])
    print(type(_data))
    if "Leaf No." in _data.columns:
        _data = _data.ffill()
        _data["Leaf No."] = _data["Leaf No."].astype(int)
    return _data


def add_leave_averages(_df: pd.DataFrame,
                       column_values_to_average: str = 'Total Chlorophyll (µg/cm2)',
                       column_to_groupby: str = "Leaf No.") -> pd.DataFrame:
    """

    Args:
        _df (pd.DataFrame): DataFrame to make the average column in
        column_values_to_average (str): name of the dataframe column of values to average
        column_to_groupby (str): name of column to average by

    Returns:
        pd.DataFrame: new dataframe with averages added to a new column.

        The new average column added with the name f"Avg {column_values_to_average}"
        of the averages of column_values_to_average

    Examples:
        >>> df = pd.DataFrame({
        ...      'Leaf No.': [1, 1, 2, 2],
        ...      'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40]
        ... })
        >>> add_leave_averages(df)
           Leaf No.  Total Chlorophyll (µg/cm2)  Avg Total Chlorophyll (µg/cm2)
        0         1                          10                            15.0
        1         1                          20                            15.0
        2         2                          30                            35.0
        3         2                          40                            35.0

    """
    # error checking the correct columns are included
    if column_to_groupby not in _df.columns:
        raise KeyError(f"'{column_to_groupby}' needs to be one of the columns in the dataframe")
    if column_values_to_average not in _df.columns:
        raise KeyError(f"Can not average '{column_values_to_average}' "
                       f"as it is not in the dataframe")

    # sometimes the Leaf number is only filled in for the first measurement not the
    # other measurements, this wil fill in the
    # column_to_groupby is in _df.columns because it was checked earlier
    _new_df = _df.ffill()
    _new_df[column_to_groupby] = _new_df[column_to_groupby].astype(int)

    # average the column_value_to_average by the column_to_groupby value
    for leaf in _new_df[column_to_groupby].unique():
        # print(f"leaf: {leaf}")
        leaf_avg = _new_df.loc[_new_df[column_to_groupby] == leaf][column_values_to_average].mean()
        # print(leaf_avg)
        _new_df.loc[_new_df[column_to_groupby] == leaf,
                    f"Avg {column_values_to_average}"] = leaf_avg
    return _new_df


if __name__ == '__main__':
    data = get_data(LEAVE)
    print(data)
