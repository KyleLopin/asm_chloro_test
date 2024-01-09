# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Visualize chlorophyll data set
"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score

# local files
import fix_outliers
from get_data import get_data

plt.style.use('bmh')
LEAVE = "Banana"
ALL_LEAVES = tuple(("Banana", "Jasmine", "Mango", "Rice", "Sugarcane"))
AXIS_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 9


# helper functions
def get_x_y(_df: pd.DataFrame, column_name: str = "Total Chlorophyll (µg/cm2)"
            ) -> tuple[pd.Series, pd.Series]:
    """ Get the x and y coordinates from a DataFrame for an average of individual measurements

    Takes a DataFrame with 2 columns, on2 with individual measurements, and another with the
    average measurements and returns the average measurements first and individual measurements
    second.

    Note the name of the individual measurements is give to column_name argument, and another
    column with the name f"Avg {column_name}" must be in the DataFrame

    Args:
        _df (pd.DataFrame): DataFrame to get data from
        column_name (str):  column name of individual measurements

    Returns:

    """
    return _df[f"Avg {column_name}"], _df[column_name]


def gauss_function(x: np.array, a: float, x0: float, sigma: float
                   ) -> np.array:
    """ Draw Gaussian function.

    This function calculates the values of a Gaussian (normal) distribution at given x values.

    The Gaussian distribution is defined as:
    f(x) = a * exp(-(x - x0)^2 / (2 * sigma^2))

    Args:
        x (np.array): Array of values to evaluate the Gaussian function.
        a (float): Amplitude or peak height of the Gaussian.
        x0 (float): Center position or mean of the Gaussian distribution.
        sigma (float): Standard deviation or spread of the Gaussian distribution.

    Returns:
        np.array: Array of values representing the Gaussian distribution,
        computed for each x in the input.

    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def get_residue_range(all_leaves: list = ALL_LEAVES) -> float:
    """ Calculate the maximum absolute residue across specified leaves after outlier removal.

    This function iterates through the specified leaves, performs outlier removal on each
    leaf's data, and calculates the maximum absolute residue across the specified leaves.
    Residues are the differences between individual measurements and their corresponding averages.

    Args:
        all_leaves (list): List of leaves for residue range calculation. Defaults to
        a global `ALL_LEAVES`.

    Returns:
        float: The maximum absolute residue across specified leaves after outlier removal.

    The function uses the `fix_outliers.remove_outliers` function to prune outliers, calculates
    residues for both the original and pruned datasets, and returns the maximum absolute residue
    observed across the specified leaves.

    """

    max_abs_residue = 0
    for leaf in all_leaves:
        leaf_df = get_data(leaf)
        leaf_df = fix_outliers.add_leave_averages(leaf_df)
        df_pruned, _, _ = fix_outliers.remove_outliers(
            leaf_df,  sigma_cutoff=3, return_std=True)
        pruned_residues = fix_outliers.calculate_residues(df_pruned)
        max_abs_residue = max(pruned_residues.max(), abs(min(pruned_residues)),
                              max_abs_residue)
        return max_abs_residue


def get_r_squared():
    pass


# visualization functions
def draw_line_between_df_pts(start_df: pd.DataFrame, end_df: pd.DataFrame, _ax: plt.Axes,
                             column_name: str = "Total Chlorophyll (µg/cm2)"):
    """ Draw a line between points common in 2 dataframes

    Take 2 DataFrames and draw a line between any 2 indexes that are the same
    in both DataFrames

    Args:
        start_df (pd.DataFrame): DataFrame from where the arrows will start
        end_df (pd.DataFrame):  DataFrame where the arrows point to
        _ax:  (pyplot.Axes):  Matplotlib Axes on which the arrows will be drawn
        column_name (str):  Name of the column with the individual measurements

    Returns:
        None
    """

    for index in start_df.index:
        x_start = start_df.loc[index][f"Avg {column_name}"]
        y_start = start_df.loc[index][column_name]
        x_end = end_df.loc[index][f"Avg {column_name}"]
        y_end = end_df.loc[index][column_name]
        # below makes a line with no arrow head
        # _ax.plot([x_start, x_end], [y_start, y_end], color='black')
        # get the total height of the y-axis to scale the arrow head
        plt_dims = _ax.get_ylim()
        plt_height = plt_dims[1] - plt_dims[0]
        _ax.arrow(x_start, y_start,
                  x_end - x_start, y_end - y_start,
                  color='black', head_width=0.015*plt_height,
                  length_includes_head=True)


def plot_histogram_residues(_df: pd.DataFrame,  ax: plt.Axes = None,
                            column_name: str = "Total Chlorophyll (µg/cm2)",
                            display_range: float = None):
    """ Plot histogram of residuals before and after outlier removal.

    This function generates a histogram of the residuals (differences between individual
    measurements and their corresponding averages) for the provided DataFrame. It also
    displays the histogram after removing outliers based on a specified sigma cutoff.

    The function adds average values if they are not present in the DataFrame and
    performs outlier removal using the `fix_outliers.remove_outliers` function,
    calculates the residuals for the pruned datasets, and .
    It then plots the histograms of the residuals,
    along with a fitted Gaussian curve to the pruned dataset.

    Args:
        _df (pd.DataFrame): Input DataFrame containing the data.
        ax (plt.Axes, optional): Matplotlib Axes to plot on. If None, a new subplot is created.
        column_name (str): The name of the column of individual measurements for which
            residuals are calculated.  The column of avera
        display_range (float, optional): The range of the x-axis for the histogram. If None, it
            is automatically determined based on the maximum absolute residual value.


    Returns:
        None

    """
    # a lot is the same as plot_individual_vs_avg
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if column_name not in _df.columns:
        raise KeyError(f"'{column_name}' not in DataFrame, you need to provide a column_name"
                       f"that is in the DataFrame")
    if f"Avg {column_name}" not in _df.columns:
        _df = fix_outliers.add_leave_averages(_df)

    df_pruned, _, _ = fix_outliers.remove_outliers(
        _df, column_name=column_name, column_sample_number="Leaf No.",
        sigma_cutoff=3, return_std=True)

    residues_pruned = fix_outliers.calculate_residues(df_pruned,
                                                      column_name=column_name)
    if display_range is None:
        display_range = max(max(residues_pruned), abs(min(residues_pruned)))
    x_bins = np.linspace(-display_range, display_range, 100)
    n, bins, _ = ax.hist(residues_pruned, bins=x_bins, color='blue', alpha=0.8,
                         label="Final data set")
    popt, _ = curve_fit(gauss_function, bins[:len(n)], n)
    gauss_fit = gauss_function(x_bins, *popt)
    ax.plot(x_bins, gauss_fit, 'r--')
    # move axis to right side and add labels
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("Bin Counts", size=AXIS_LABEL_SIZE)
    # make xlabel, first get the units used from the column name which
    # should have the units at the end in between ( )'s
    measurement_unit = f"({column_name.split('(')[1]}"
    print(measurement_unit)
    ax.set_xlabel(f"Measurement Residue {measurement_unit}", size=AXIS_LABEL_SIZE)


def plot_individual_vs_avg(_df: pd.DataFrame,  ax: plt.Axes = None,
                           column_name: str = "Total Chlorophyll (µg/cm2)"):
    """  Scatter plot the individual measurements versus the average measurements of samples

    To view the relationship between individual measurements and the averages of the measurements,
    and to view any outliers, plot the individual measurements (y-axis) and
    the average measurement (x-axis).  The remove_outliers function will be called from the
    fix_outliers.py file and remove any measurements over 3 sigma from the average and display
    and highlight those measurements on the graph.

    Args:
        _df (pd.DataFrame): DataFrame containing the data to be displayed.
        ax (Optional[plt.Axes]): Matplotlib Axes on which the plot will be displayed.
                                 If None, a new subplot will be created. Defaults to None.
        column_name (str): Name of the column containing individual measurements.

    Returns:
        None
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if column_name not in _df.columns:
        raise KeyError(f"'{column_name}' not in DataFrame, you need to provide a column_name"
                       f"that is in the DataFrame")
    if f"Avg {column_name}" not in _df.columns:
        raise KeyError(f"'Avg {column_name}' not in DataFrame, need to have a column with "
                       f"the name 'Avg [column_name]' that holds the averages")
    # x, y = get_x_y(_df, column_name=column_name)

    # get outliers
    df_pruned, outlier_idxs, std = fix_outliers.remove_outliers(
        _df, column_name=column_name, column_sample_number="Leaf No.",
        sigma_cutoff=3, return_std=True)
    # put an X on the outliers
    x_outliers, y_outliers = get_x_y(_df.loc[outlier_idxs])
    num_outliers = len(outlier_idxs)
    # get outlier sample numbers
    outlier_numbers = _df.loc[outlier_idxs]["Leaf No."].unique()
    # get the samples that are with the outliers
    _df_co_samples_of_outliers = _df[_df["Leaf No."].isin(outlier_numbers)]
    # and the samples x, y coords
    x_co_samples, y_co_samples = get_x_y(_df_co_samples_of_outliers)
    x_pruned, y_pruned = get_x_y(df_pruned)
    # get number of data points to display in legend
    num_data_pts = len(df_pruned.index)
    # make legend
    ax.scatter(x_pruned, y_pruned, marker='o', c='blue', s=20, alpha=0.7,
               lw=1, label=f"Final data set:\n{num_data_pts} data points")
    ax.scatter(x_co_samples, y_co_samples, marker='o', c='green', s=20, alpha=0.4,
               lw=1, label="samples of outliers")
    ax.scatter(x_outliers, y_outliers, marker='x', c='red', s=50, alpha=0.7,
               lw=1, label=f"{num_outliers} removed outliers")
    print(f"std: {std}")
    print(_df_co_samples_of_outliers.drop(outlier_idxs))
    draw_line_between_df_pts(_df_co_samples_of_outliers.drop(outlier_idxs), df_pruned,
                             ax, column_name=column_name)
    ax.set_xlabel(f"Average {column_name}", size=AXIS_LABEL_SIZE)
    ax.set_ylabel(f"Individual {column_name}", size=AXIS_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    dispaly_r2(_df, ax, column_name)


def dispaly_r2(original_df: pd.DataFrame, ax: plt.Axes,
               column_name: str):
    residues = fix_outliers.calculate_residues()


def plot_both_leaf_graphs(_df: pd.DataFrame, axes: tuple[plt.Axes, plt.Axes] = None,
                          column_name: str = "Total Chlorophyll (µg/cm2)",
                          max_range: float = None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    plot_individual_vs_avg(_df, axes[0], column_name=column_name)
    plot_histogram_residues(_df, axes[1], column_name=column_name,
                            display_range=max_range)


if __name__ == '__main__':
    residue_range = get_residue_range()
    print(residue_range)
    data = get_data(LEAVE)
    print(data)
    data = fix_outliers.add_leave_averages(data)
    print(data)
    # plot_individual_vs_avg(data)
    # plot_histogram_residues(data, range=residue_range)
    plot_both_leaf_graphs(data, max_range=residue_range)
    plt.show()
