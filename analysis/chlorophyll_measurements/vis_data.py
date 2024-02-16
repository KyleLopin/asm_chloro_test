# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Visualize chlorophyll data set
"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
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
ALL_LEAVES = tuple(("Mango", "Banana", "Jasmine", "Rice", "Sugarcane"))
AXIS_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 9
AX_TITLE_SIZE = 12
R2_ANNOTATE_POSITION = (.55, .08)
MAE_ANNOTATE_POSITION = (.53, .8)
FIGURE_LABEL_ANNOTATE_POSITION_L = (0.0, 1.05)
FIGURE_LABEL_ANNOTATE_POSITION_R = (0.02, .88)
BIN_COUNTS = [0, 10, 20, 30, 40]
BIN_Y_LIM = [0, 45]


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
        # print(f"leaf: {leaf} has range {max_abs_residue}")
    return max_abs_residue


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
                            display_range: float = None,
                            title: str = ""):
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
        title (str): Title to display on the graph, if empty string no title will be
            displayed

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
        display_range = get_residue_range()
    x_bins = np.linspace(-display_range, display_range, 100)
    n, bins, _ = ax.hist(residues_pruned, bins=x_bins, color='blue', alpha=0.8,
                         label="Final data set")
    popt, _ = curve_fit(gauss_function, bins[:len(n)], n)
    gauss_fit = gauss_function(x_bins, *popt)
    ax.plot(x_bins, gauss_fit, 'r--')
    print(f"max bin value: {max(n)}")  # get max bin count for all samples
    # move axis to right side and add labels, or not
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # set bin counts on y label as integers
    ax.set(yticks=BIN_COUNTS)
    ax.set_ylim(BIN_Y_LIM)
    # ax.set_ylabel("Bin Counts", size=AXIS_LABEL_SIZE)
    # make xlabel, first get the units used from the column name which
    # should have the units at the end in between ( )'s
    # measurement_unit = f"({column_name.split('(')[1]}"
    # ax.set_xlabel(f"Measurement Residue {measurement_unit}", size=AXIS_LABEL_SIZE)
    if title:
        ax.set_title(title, size=AX_TITLE_SIZE)
    display_mea(_df, ax, column_name=column_name)


def plot_individual_vs_avg(_df: pd.DataFrame,  ax: plt.Axes = None,
                           column_name: str = "Total Chlorophyll (µg/cm2)",
                           title: str = ""):
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
        title (str): Title to display on the graph, if empty string no title will be
            displayed.

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

    # make line for reference at the beginning, so it is in the back
    data_range = [min(_df[column_name]), max(_df[column_name])]
    ax.plot(data_range, data_range, color='black', linestyle='--')

    # get outliers
    df_pruned, outlier_idxs, _ = fix_outliers.remove_outliers(
        _df, column_name=column_name, column_sample_number="Leaf No.",
        sigma_cutoff=3, return_std=True)
    print(f"Pruned size: {df_pruned.shape}")
    print(f"removed idxs: {outlier_idxs}")
    # put an X on the outliers
    x_outliers, y_outliers = get_x_y(_df.loc[outlier_idxs], column_name=column_name)
    # get outlier sample numbers
    outlier_numbers = _df.loc[outlier_idxs]["Leaf No."].unique()
    # get the samples that are with the outliers
    _df_co_samples_of_outliers = _df[_df["Leaf No."].isin(outlier_numbers)]
    # and the samples x, y coords
    x_co_samples, y_co_samples = get_x_y(_df_co_samples_of_outliers, column_name=column_name)
    x_pruned, y_pruned = get_x_y(df_pruned, column_name=column_name)
    # get number of data points to display in legend if used
    # num_data_pts = len(df_pruned.index)

    ax.scatter(x_pruned, y_pruned, marker='o', c='blue', s=20, alpha=0.7,
               lw=1, label="Final data set")
    ax.scatter(x_co_samples, y_co_samples, marker='o', c='green', s=20, alpha=0.4,
               lw=1, label="samples of outliers")
    ax.scatter(x_outliers, y_outliers, marker='x', c='red', s=50, alpha=0.7,
               lw=1, label="removed outliers")
    draw_line_between_df_pts(_df_co_samples_of_outliers.drop(outlier_idxs), df_pruned,
                             ax, column_name=column_name)
    # uncomment for single use, take out to make the last big figure
    # ax.scatter(x_pruned, y_pruned, marker='o', c='blue', s=20, alpha=0.7,
    #            lw=1, label=f"Final data set:\n{num_data_pts} data points")
    # ax.scatter(x_co_samples, y_co_samples, marker='o', c='green', s=20, alpha=0.4,
    #            lw=1, label="samples of outliers")
    # ax.scatter(x_outliers, y_outliers, marker='x', c='red', s=50, alpha=0.7,
    #            lw=1, label=f"{num_outliers} removed outliers")
    # ax.set_xlabel(f"Average {column_name}", size=AXIS_LABEL_SIZE)
    # ax.set_ylabel(f"Individual {column_name}", size=AXIS_LABEL_SIZE)
    # ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=False)
    # display r squared for data
    display_r2(_df, ax, column_name=column_name)
    if title:  # display title if one is given
        # put the title on the right side
        ax.set_title(title, size=AX_TITLE_SIZE, loc='right',
                     fontweight='bold')


# helper function to display annotations on the graphs
def display_r2(original_df: pd.DataFrame, ax: plt.Axes,
               column_name: str = "Total Chlorophyll (µg/cm2)"):
    """ Calculate and display r squared for the data.

    Calculate the r squared for the individual samples versus the sample average measurements
    for all the data, and the r squared after removing 3 sigma data.  Uses get_x_y to get the
    data to calculate the r squared from and fix_outliers.remove_outliers to remove the outliers.

    Use constant R2_ANNOTATE_POSITION to change position.

    Args:
        original_df (pd.DataFrame): DataFrame to calculate r squared from
        ax (plt.Axes): axes put annotation on
        column_name (str): Name of the column containing individual measurements.

    Returns:
        None
    """
    x_original, y_original = get_x_y(original_df, column_name=column_name)
    r2_original = r2_score(x_original, y_original)
    df_pruned, _ = fix_outliers.remove_outliers(
        original_df,  sigma_cutoff=3)
    x_pruned, y_pruned = get_x_y(df_pruned, column_name=column_name)
    r2_pruned = r2_score(x_pruned, y_pruned)
    r2_string = f"r\u00B2 original = {r2_original:.3f}\n" \
                f"r\u00B2 final = {r2_pruned:.3f}"
    ax.annotate(r2_string, R2_ANNOTATE_POSITION, xycoords='axes fraction')


def display_mea(original_df: pd.DataFrame, ax: plt.Axes,
                column_name: str = "Total Chlorophyll (µg/cm2)"):
    """ Display the mean absolute error on the graph.
    A lot of copy and paste from display_r2, but whatever
    Use constant R2_ANNOTATE_POSITION to change position.

    Args:
        original_df (pd.DataFrame): DataFrame to calculate r squared from
        ax (plt.Axes): axes put annotation on
        column_name (str): Name of the column containing individual measurements.

    Returns:
        None
    """
    x_original, y_original = get_x_y(original_df, column_name=column_name)
    mae_original = mean_absolute_error(x_original, y_original)
    df_pruned, _ = fix_outliers.remove_outliers(
        original_df, sigma_cutoff=3)
    x_pruned, y_pruned = get_x_y(df_pruned, column_name=column_name)
    mae_pruned = mean_absolute_error(x_pruned, y_pruned)
    mae_string = f"MAE original = {mae_original:.2f}\n" \
                 f"MAE final = {mae_pruned:.2f}"
    ax.annotate(mae_string, MAE_ANNOTATE_POSITION, xycoords='axes fraction')


def plot_both_leaf_graphs(_df: pd.DataFrame, axes: tuple[plt.Axes, plt.Axes] = None,
                          column_name: str = "Total Chlorophyll (µg/cm2)",
                          max_range: float = None):
    """ Combine the plot_individual_vs_avg and plot_histogram_residues functions and
    put them side by side.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    plot_individual_vs_avg(_df, axes[0], column_name=column_name)
    plot_histogram_residues(_df, axes[1], column_name=column_name,
                            display_range=max_range,
                            title="Histogram of residues of measured\n"
                                  "chlorophyll levels after removing outliers")


def plot_all_leaves(column_name: str = "Total Chlorophyll (µg/cm2)"):
    """ Make final figure to show chlorophyll levels"""
    _, axes = plt.subplots(5, 2, figsize=(7.5, 10))
    max_residue_range = get_residue_range()
    print(f"max range: {max_residue_range}")
    _use_columns = ["Total Chlorophyll (µg/cm2)", "Spot", "Leaf No."]
    if column_name not in _use_columns:
        _use_columns.append(column_name)
    for row, leaf in enumerate(ALL_LEAVES):
        print(f"leaf: {leaf}")
        data = get_data(leaf, use_columns=_use_columns)  # get the data
        # and the average column
        data = fix_outliers.add_leave_averages(data, column_values_to_average=column_name)
        # plot the individual vs average plots
        plot_individual_vs_avg(data, axes[row][0], column_name=column_name,
                               title=f"{leaf} leaves")
        # make figure numbers, i+1 for each row and (a) (b) for columns
        axes[row][0].annotate(f"{row+1} a)", FIGURE_LABEL_ANNOTATE_POSITION_L,
                              xycoords='axes fraction', fontsize=12)
        # if row != 0:  # the histogram title is on the top
        axes[row][1].annotate("b)", FIGURE_LABEL_ANNOTATE_POSITION_R,
                              xycoords='axes fraction', fontsize=12)

        # plot histograms
        if row == 0:  # add a title to the first graph
            plot_histogram_residues(data, axes[row][1],
                                    display_range=max_residue_range,
                                    column_name=column_name,
                                    title="Histogram of residues of measured  \n"
                                          "chlorophyll levels after removing outliers  ")
        else:
            plot_histogram_residues(data, axes[row][1],
                                    display_range=max_residue_range)

    # ==== CONFIGURE LEFT GRAPHS =========
    # put legend on only the top graph,
    # mango is the best one to use as it has not points in the area
    axes[0][0].legend(fontsize=LEGEND_FONT_SIZE, frameon=False,
                      loc='upper left')
    # set y-label only on the middle graph
    axes[2][0].set_ylabel("Individual Total Chlorophyll (µg/cm2)", size=AXIS_LABEL_SIZE)
    axes[2][0].set_ylabel("Bin Counts", size=AXIS_LABEL_SIZE)
    # put the x-label on the bottom
    axes[4][0].set_xlabel("Average Total Chlorophyll (µg/cm2)", size=AXIS_LABEL_SIZE)
    # put the x-label on the bottom of histogram graphs
    measurement_unit = f"({column_name.split('(')[1]}"  # get unit being measure for x label
    axes[4][1].set_xlabel(f"Measurement Residue {measurement_unit}", size=AXIS_LABEL_SIZE)


if __name__ == '__main__':
    # residue_range = get_residue_range()
    # print(residue_range)
    # data = get_data(LEAVE)
    # print(data)
    # data = fix_outliers.add_leave_averages(data)
    # print(data)
    # plot_individual_vs_avg(data)
    # plot_histogram_residues(data, range=residue_range)
    # plot_both_leaf_graphs(data, max_range=residue_range)
    plot_all_leaves(column_name='Chlorophyll b (µg/cm2)')
    plt.tight_layout()
    plt.show()
    # plt.savefig("chlorophyll_r2_chl_a.svg")
