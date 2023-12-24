# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
chlorophyll_analysis.py

This script provides functions for analyzing chlorophyll data, including loading data, removing outliers,
and adding average values to the dataset.

Dependencies:
- numpy as np
- pandas as pd
- scipy.stats.zscore
- get_data.py (local file)

Functions:
1. `add_leave_averages(_df: pd.DataFrame, column_values_to_average: str = 'Total Chlorophyll (µg/cm2)',
                       column_to_groupby: str = "Leaf No.") -> pd.DataFrame`:
   Add a column of average values to a DataFrame, overwriting an existing column if it exists.

2. `gauss_function(x_line: np.array, height: float, center: float, sigma: float) -> np.array`:
   Create a Gaussian distribution based on given parameters.

3. `remove_outliers_recursive(_df: pd.DataFrame, column_name: str = 'Total Chlorophyll (µg/cm2)',
                              column_sample_number: str = "Leaf No.",
                              sigma_cutoff: float = 3.0) -> tuple[pd.DataFrame, list]`:
   Remove outliers from individual measurements from their average recursively until all z-scores are below the cutoff.

4. `remove_outliers(_df: pd.DataFrame, column_name: str = 'Total Chlorophyll (µg/cm2)',
                    column_sample_number: str = "Leaf No.", sigma_cutoff: float = 3.0) -> tuple[pd.DataFrame, list]`:
   Remove outliers from individual measurements based on their average until all z-scores are below the cutoff.

5. `drop_measurement_w_sample_check(_df: pd.DataFrame, idx_to_remove: int,
                                    sample_number_column: str = "Leaf No.",
                                    required_number_samples: int = 2) -> tuple[pd.DataFrame, list]`:
   Remove one sample by index from the DataFrame and check that the required number of samples is still present.

Usage:
   If executed as a standalone script, it loads chlorophyll data, removes outliers, and prints the removed indices
   along with the corresponding leaf data.
"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import numpy as np
import pandas as pd
from scipy.stats import zscore

# local files
from get_data import get_data

LEAVE = "Banana"


def add_leave_averages(_df: pd.DataFrame,
                       column_values_to_average: str = 'Total Chlorophyll (µg/cm2)',
                       column_to_groupby: str = "Leaf No.") -> pd.DataFrame:
    """  Add a column of average values to a DataFrame, WILL OVERWRITE AN EXISTING COLUMN

    Take a DataFrame with a column of individual measurements (column_values_to_average)
    and their sample numbers (column_to_groupby) and calculate a new column with each samples
    average value in a new column f"Avg {column_values_to_average}" that will be overwritten if
    it exists on the inputted DataFrame

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
        raise KeyError(f"'{column_to_groupby}' needs to be one of the columns in the dataframe,"
                       f" or the 'column_to_groupby' has to be supplied")
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


def remove_outliers_recursive(_df: pd.DataFrame,
                              column_name: str = 'Total Chlorophyll (µg/cm2)',
                              column_sample_number: str = "Leaf No.",
                              sigma_cutoff: float = 3.0) -> tuple[pd.DataFrame, list]:
    """ Remove outliers of individual measurements from their average recursively.

    Take a DataFrame that has a column with multiple measurements of the same sample and a column
    with the column that labels which samples each measurement are for (column_sample_number),
    calculate the difference between each individual measurement with the
    average(residual), take the z-score of each residual and remove the sample with the largest
    z-score and repeat till all measurements have a z-score below the sigma_cutoff.
    The average of each sample is updated in each iteration to account for changes caused
    by removing outliers. The removed indexes are stored and returned
    along with the updated DataFrame.

    Args:
        _df (pd.DataFrame): Input DataFrame containing the data.
        column_name (str): The name of the column of the individual measurements for which
        outliers are to be removed.
        column_sample_number (str): The column used to show which measurements are from the
         same sample.
        sigma_cutoff (float, optional): The cutoff value in terms of standard deviations
        for considering data points as outliers. Defaults to 3.

    Returns:
        - pd.DataFrame: The DataFrame with outliers removed after each iteration and a new
        column with the averages of the column_name based on column_sample_number with the column
        name of "Avg {column_name}".
        - list: A list of indexes corresponding to the removed outliers in each iteration.

    """
    if column_sample_number not in _df.columns:
        raise KeyError(f"'{column_sample_number}' needs to be one of the columns in the dataframe,"
                       f" or the 'column_to_groupby' has to be supplied")
    removed_indexes = []  # to store the indexes of the removed data
    while True:  # loop till the break condition returns the final data frame
        # need to get the average of each sample every time in the iteration as removing outlying
        # measurements will change the average for the samples
        _df = add_leave_averages(_df, column_values_to_average=column_name,
                                 column_to_groupby=column_sample_number)

        # calculate the residues of the column from their average
        values = _df[column_name]
        average_values = _df[f"Avg {column_name}"]
        residues = values - average_values  # type: pd.Series
        print(residues.std())
        # get the z_scores of the indicated column
        z_scores = zscore(residues).abs()  # type: pd.Series
        # go through the z_scores and remove the largest one, saving the indexes
        # you can not just remove all z_score more than the cutoff because 1
        # outlier in a series can affect the z_score of the other measurements
        print(f"z max: {z_scores.max()}, max idx: {z_scores.idxmax()}")
        if abs(z_scores.max()) > sigma_cutoff:
            _df, _idx = drop_measurement_w_sample_check(_df, z_scores.idxmax(),
                                                        sample_number_column=column_sample_number)
            removed_indexes.extend(_idx)
            # removed_indexes.append(z_scores.idxmax())
            # _df.drop(labels=z_scores.idxmax(), inplace=True)
        else:
            return _df, removed_indexes


def remove_outliers(_df: pd.DataFrame,
                    column_name: str = 'Total Chlorophyll (µg/cm2)',
                    column_sample_number: str = "Leaf No.",
                    sigma_cutoff: float = 3.0, return_std: bool = False,
                    ) -> (tuple[pd.DataFrame, list] |
                          tuple[pd.DataFrame, list, float]):
    """ Remove outliers of individual measurements from their average.

    Take a DataFrame that has a column with multiple measurements of the same sample and a column
    with the label of the samples for each measurement (column_sample_number),
    calculate the difference between each individual measurement with the
    average (i.e. the residual).  The standard deviation of the original residues is calculated
    and the

    Args:
        _df (pd.DataFrame): Input DataFrame containing the data.
        column_name (str): The name of the column of the individual measurements for which
        outliers are to be removed.
        column_sample_number (str): The column used to show which measurements are from the
         same sample.
        sigma_cutoff (float, optional): The cutoff value in terms of standard deviations
        for considering data points as outliers. Defaults to 3.

    Returns:
        - pd.DataFrame: The DataFrame with outliers removed after each iteration and a new
        column with the averages of the column_name based on column_sample_number with the column
        name of "Avg {column_name}".
        - list: A list of indexes corresponding to the removed outliers in each iteration.

    """
    original_residue_std = None  # will work to save state for when the algorithm is first run

    if column_sample_number not in _df.columns:
        raise KeyError(f"'{column_sample_number}' needs to be one of the columns in the dataframe,"
                       f" or the 'column_to_groupby' has to be supplied")
    removed_indexes = []  # to store the indexes of the removed data
    # loop through and calculate the averages, calculate the difference between the individual
    # measurements and the average measurement (residue), calculate the standard deviation
    # of the residues (the mean should be zero by tautology), calculate the z-scores from the
    # standard deviation (only use the original standard deviation for all iterations),
    # remove the largest absolute z-score until all calculated z-scores are below the threshold
    while True:
        # add the averages for each sample, will overwrite if needed
        _df = add_leave_averages(_df, column_values_to_average=column_name,
                                 column_to_groupby=column_sample_number)
        # calculate the residues of the column from their average
        values = _df[column_name]
        average_values = _df[f"Avg {column_name}"]
        residues = values - average_values  # type: pd.Series
        # get the z_scores of the indicated column
        if not original_residue_std:
            # get the original standard deviation and check it calculated the z-scores correctly
            z_scores_from_stats = zscore(residues, ddof=1)  # type: pd.Series
            original_residue_std = residues.std()
            z_scores_homemade = residues / original_residue_std
            pd.testing.assert_series_equal(z_scores_from_stats, z_scores_homemade)  # make sure z
        else:  # optional but prevent recalculating 2 times the first loop
            z_scores_homemade = residues / original_residue_std
        z_scores_homemade = z_scores_homemade.abs()
        if z_scores_homemade.max() > sigma_cutoff:
            _df, _idx = drop_measurement_w_sample_check(_df, z_scores_homemade.idxmax(),
                                                        sample_number_column=column_sample_number)
            removed_indexes.extend(_idx)
            # _df.drop(labels=z_scores_homemade.idxmax(), inplace=True)
        else:
            if return_std:
                return _df, removed_indexes, original_residue_std
            else:
                return _df, removed_indexes


def drop_measurement_w_sample_check(_df: pd.DataFrame, idx_to_remove: int,
                                    sample_number_column: str = "Leaf No.",
                                    required_number_samples: int = 2) -> tuple[pd.DataFrame, list]:
    """ Remove 1 sample with the index idx_to_remove from the _df DataFrame and check that
    a required number of samples are still present.

    After the index is removed the sample_number_column of the removed index is checked if the
    of samples with the same value in the column_name is less than required_number_samples,
    after removal, all samples with the same column value will be removed.

    Args:
        _df (pd.DataFrame): DataFrame to drop a row from
        idx_to_remove (int): index to remove from the DataFrame
        sample_number_column (str): name of the column that has the sample names / numbers that
        group each individual measurements
        required_number_samples (int): number of individual measurement

    Returns:
        pd.DataFrame: new DataFrame with the samples removed
        list: list of index / indices removed

    """
    removed_sample_name = _df[sample_number_column][idx_to_remove]
    _df_of_removed_sample_names = _df[_df[sample_number_column] == removed_sample_name]
    # calculate number of samples with same name, google says this is the fastest way
    number_samples = len(_df_of_removed_sample_names.index)
    if (number_samples-1) >= required_number_samples:  # -1 because will be 1 less after drop
        return _df.drop([idx_to_remove]), [idx_to_remove]
    # else remove all samples of same name and get indexes
    return (_df.drop(_df_of_removed_sample_names.index),
            list(_df_of_removed_sample_names.index.values))


if __name__ == '__main__':
    data = get_data(LEAVE)
    # data = add_leave_averages(data)
    pruned_df, removed_idx = remove_outliers(data)

    print(removed_idx)
    # figure out what leaves are removed to see if 1 leave had 2 outliers that got removed
    print(data.iloc[removed_idx])
    # for leaves in data removed, print out each and check
    for idx in removed_idx:
        leaf_number = data.iloc[idx]["Leaf No."]
        print(f"leaf {leaf_number} data, index {idx} removed")
        print(data[data["Leaf No."] == leaf_number])
