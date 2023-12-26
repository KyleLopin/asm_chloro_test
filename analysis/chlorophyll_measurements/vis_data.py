# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Visualize chlorophyll data set
"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import matplotlib.pyplot as plt
import pandas as pd

# local files
import fix_outliers
from get_data import get_data

plt.style.use('bmh')
LEAVE = "Banana"


def get_x_y(_df: pd.DataFrame, column_name: str = "Total Chlorophyll (µg/cm2)"
            ) -> tuple[pd.Series, pd.Series]:
    return _df[f"Avg {column_name}"], _df[column_name]


def draw_line_between_df_pts(start_df: pd.DataFrame, end_df: pd.DataFrame, _ax:plt.Axes,
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
        plt_dims= _ax.get_ylim()
        plt_height = plt_dims[1] - plt_dims[0]
        _ax.arrow(x_start, y_start,
                  x_end - x_start, y_end - y_start,
                  color='black', head_width=0.015*plt_height,
                  length_includes_head=True)


def plot_individual_vs_avg(_df: pd.DataFrame,  ax: plt.Axes = None,
                           column_name: str = "Total Chlorophyll (µg/cm2)"):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
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
    # x_outliers = _df.loc[outlier_idxs][f"Avg {column_name}"]
    # y_outliers = _df.loc[outlier_idxs][column_name]
    x_outliers, y_outliers = get_x_y(_df.loc[outlier_idxs])
    num_outliers = len(outlier_idxs)
    # get outlier sample numbers
    outlier_numbers = _df.loc[outlier_idxs]["Leaf No."].unique()
    print(outlier_numbers)
    _df_co_samples_of_outliers = _df[_df["Leaf No."].isin(outlier_numbers)]
    print(_df_co_samples_of_outliers)
    x_co_samples, y_co_samples = get_x_y(_df_co_samples_of_outliers)
    print(f"type x_co: {type(x_co_samples)}")
    x_pruned = df_pruned[f"Avg {column_name}"]
    y_pruned = df_pruned[column_name]
    num_data_pts = len(df_pruned.index)
    ax.scatter(x_pruned, y_pruned, marker='o', c='blue', s=20, alpha=0.8,
               lw=1, label=f"Final data set: {num_data_pts} data points")
    ax.scatter(x_co_samples, y_co_samples, marker='o', c='green', s=20, alpha=0.5,
               lw=1, label="samples of outliers")
    ax.scatter(x_outliers, y_outliers, marker='x', c='red', s=50, alpha=0.8,
               lw=1, label=f"{num_outliers} removed outliers")
    print(f"std: {std}")
    print(_df_co_samples_of_outliers.drop(outlier_idxs))
    draw_line_between_df_pts(_df_co_samples_of_outliers.drop(outlier_idxs), df_pruned,
                             ax, column_name=column_name)
    ax.legend()


if __name__ == '__main__':
    data = get_data(LEAVE)
    print(data)
    data = fix_outliers.add_leave_averages(data)
    print(data)
    plot_individual_vs_avg(data)
    plt.show()
