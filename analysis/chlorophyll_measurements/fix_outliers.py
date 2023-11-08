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
import numpy as np

import pandas as pd
from scipy.stats import zscore


plt.style.use('bmh')

LEAVE = "Banana"

DATA_FOLDER = Path.cwd().parent.parent / "data" / "chlorophyll_data" / "collected_data"
print(DATA_FOLDER)


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
    print(_data.columns)
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


def gauss_function(x_line: np.array, height: float,
                   center: float, sigma: float) -> np.array:
    """ Create a Gaussian distribution.

    Take a list or array of numbers and plot the Gaussian (normal) distribution along
    the given array of numbers.

    Args:
        x_line (numpy.array, shape=(N,): Variables to evaluate the Gaussian function.
        height (float): Peak height of the Gaussian.
        center (float): Center position of the distribution
        sigma (float): Standard deviation (variance) of the Gaussian distribution.

    Returns:
        numpy.array: An array of values representing the Gaussian distribution,
        computed for each x in x_line.

    The Gaussian distribution is defined as:
    f(x) = height * exp(-(x - center)^2 / (2 * sigma^2))

    """
    return height*np.exp(-(x_line - center) ** 2 / (2 * sigma ** 2))


def remove_outliers(_df: pd.DataFrame,
                    column_name: str = 'Total Chlorophyll (µg/cm2)',
                    sigma_cutoff: float = 3.0) -> pd.DataFrame:
    if f"Avg {column_name}" not in _df.columns:
        raise KeyError(f"The average of {column_name} must also be in the dataframe with"
                       f"the columns name: 'Avg {column_name}'")
    while True:  # loop till the break condition returns the final data frame
        # calculate the residues of the column from their average
        values = _df[column_name]
        average_values = _df[f"Avg {column_name}"]
        residues = values - average_values  # type: pd.Series
        # get the z_scores of the indicated column
        z_scores = zscore(residues)  # type: pd.Series
        # go through the z_scores and remove the largest one, saving the indexes
        # you can not just remove all z_score more than the cutoff because 1
        # outlier in a series can affect the z_score of the other measurements
        removed_indexes = []  # TODO: impliment this
        if abs(z_scores.max()) < sigma_cutoff:
            _df.drop(labels=z_scores.idxmax(), inplace=True)
            return _df



if __name__ == '__main__':
    data = get_data(LEAVE)
    data = add_leave_averages(data)
    print(data)
    remove_outliers(data)
