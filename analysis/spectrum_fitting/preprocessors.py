# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
polynomial_expansion.py

This module provides a function for expanding the features of a
DataFrame by creating polynomial features.

Functions:
    polynomial_expansion: Expand the features of the input DataFrame
    by creating polynomial features up to the specified degree.

Usage:
    from polynomial_expansion import polynomial_expansion
    expanded_df = polynomial_expansion(input_df, degree=2)
"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import matplotlib.pyplot as plt  # for visualizing data
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# local files
import get_data  # for visualizing data


def polynomial_expansion(x: pd.DataFrame,
                         degree: int = 2,
                         standerdize: bool = False
                         ) -> pd.DataFrame:
    """
    Expand the features of the input DataFrame `x` by creating polynomial features
    up to the specified `degree`.

    Args:
        x (pd.DataFrame): Input DataFrame containing the features.
        degree (int, optional): Degree of polynomial features to be created. Defaults to 2.
        standerdize (bool, optional): If a StandardScalar should be applied before the
        expanding the data.  Defaults to False

    Returns:
        pd.DataFrame: DataFrame with polynomial features added.

    Raises:
        ValueError: If `degree` is less than 2.

    Examples:
        >>> import pandas as pd
        >>> data = {'x1': [1, 2, 3], 'x2': [4, 5, 6]}
        >>> df = pd.DataFrame(data)
        >>> polynomial_expansion(df, degree=2)
           x1  x2  (x1)^2  (x2)^2
        0   1   4       1      16
        1   2   5       4      25
        2   3   6       9      36

        >>> polynomial_expansion(df, degree=3)
           x1  x2  (x1)^2  (x2)^2  (x1)^3  (x2)^3
        0   1   4       1      16       1      64
        1   2   5       4      25       8     125
        2   3   6       9      36      27     216

        >>> polynomial_expansion(df, degree=1)
        Traceback (most recent call last):
        ...
        ValueError: Input degrees needs to be 2 or larger, 1 is not valid
    """
    if degree < 2:
        raise ValueError(f"Input degrees needs to be 2 or larger, {degree} is not valid")
    print(x)
    if standerdize:
        x_scaled = StandardScaler().fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=x.columns)

    new_x = x.copy()

    for power in range(2, degree+1):
        for column in x.columns:
            new_x[f"({column})^{power}"] = x[column]**power
    return new_x


def snv(input_data) -> pd.DataFrame:
    """
    Standard Normal Variate function to preprocess a pandas DataFrame.

    Core code taken from:
    https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/

    Args:
        input_data (pandas.DataFrame or numpy.ndarray): Data to be corrected.

    Returns:
        pandas.DataFrame or numpy.ndarray: SNV corrected data. If a DataFrame was passed in,
        returns a DataFrame, otherwise, returns a numpy array.
    """
    _type = type(input_data)
    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    _columns = None
    if type(input_data) is pd.DataFrame:
        _columns = input_data.columns
        _index = input_data.index
        input_data = input_data.to_numpy()

    for i in range(input_data.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    if _type is pd.DataFrame:
        return pd.DataFrame(data_snv, columns=_columns, index=_index)
    # else return a numpy array
    return data_snv


def msc(input_data: pd.DataFrame, reference: pd.Series=None) -> (
        tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]):
    """
    Perform Multiplicative Scatter Correction.

    Reference: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/

    Args:
        input_data (pandas.DataFrame or numpy.ndarray): Data to be corrected.
        reference (numpy.ndarray, optional): Reference spectrum. If not given,
        it's estimated from the mean.

    Returns:
        Tuple[numpy.ndarray or pandas.Series, numpy.ndarray or pandas.Series]:
            MSC corrected data and reference spectrum. If input_data is a DataFrame,
            returns DataFrames, else returns numpy arrays.
    """
    # mean centre correction
    _type = type(input_data)

    if _type is pd.DataFrame:
        _columns = input_data.columns
        _index = input_data.index
        input_data = input_data.to_numpy()
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()
    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)
        # Apply correction
        data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
    if _type is pd.DataFrame:
        return pd.DataFrame(data_msc, columns=_columns,
                            index=_index), pd.Series(ref)

    return data_msc, ref


def recursive_feature_elimination(x: pd.DataFrame, y: pd.Series,
                                  regr: RegressorMixin = LinearRegression(),
                                  display_results: bool = True):
    pass


if __name__ == '__main__':
    # _x = pd.DataFrame([[1, 2], [3, 4]], columns=["450 nm", "500 nm"])
    #
    # _x = polynomial_expansion(_x)
    # print('===')
    # print(_x)
    x, _, _ = get_data.get_x_y(sensor="as7262",  # led = "b'White'",
                               leaf="mango",
                               measurement_type="raw",
                               int_time=150,
                               send_leaf_numbers=True)
    x, _ = msc(x)
    plt.plot(x.T)
    plt.show()
