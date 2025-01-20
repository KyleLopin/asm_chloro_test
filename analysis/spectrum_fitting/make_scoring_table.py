# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
This module provides functions for evaluating regression models and generating heatmaps
for validation scores such as R² and MAE. The primary functionality includes loading
sensor data, fitting models, calculating performance metrics, and visualizing results
through heatmaps and grouped bar charts.

Functions:
- create_validation_heatmaps: Loads validation scores from an Excel file and generates
  heatmaps for R² and MAE scores.
- make_table: Evaluates regression models across different sensors and leaves, calculates
  R² and MAE scores, and generates heatmaps for the results.
- plot_heatmaps: Generates heatmaps for R² and MAE values, either displaying them or
  saving to a file.
- plot_grouped_bar_charts: Creates grouped bar charts displaying R² and MAE scores for
  different leaves and sensor types.

Dependencies:
- matplotlib, numpy, pandas, seaborn, sklearn

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# local files
import get_data
import helper_functions
import remove_outliers
SENSORS = ["as7262", "as7263", "as7265x"]
LEAVES = ["banana", "jasmine", "mango", "rice", "sugarcane"]
pd.set_option('display.max_rows', 500)
N_SPLITS = 200  # set to 100 for low variance, 10 to be quick

pls_n_comps = {"as7262": {"banana": 6, "jasmine": 4, "mango": 5, "rice": 6, "sugarcane": 6},
               "as7263": {"banana": 5, "jasmine": 5, "mango": 5, "rice": 5, "sugarcane": 5},
               "as7265x": {"banana": 8, "jasmine": 14, "mango": 11, "rice": 6, "sugarcane": 6}}


def create_validation_heatmaps(filename: str = "") -> None:
    """
    Load validation scores from an Excel file and create heatmaps for R² and MAE scores.

    Parameters
    ----------
    filename : str, optional
        The name of the file to save the generated heatmaps as a PDF. If
        a falsy value (e.g., an empty string), the heatmaps will not be saved (default is "").

    Returns
    -------
    None
        Displays the heatmaps.
        Optionally saves them to the specified file if `filename` is provided.

    Notes
    -----
    - The function reads data from an Excel file named "data_for_tables.xlsx",
      located in a "data" folder, in the "validation scores" sheet.
    - Uses plot_heatmaps to display and optionally save file.
    """
    # Load data from the Excel file
    # Construct the file path
    base_path = Path(__file__).resolve().parents[2] / "data"
    file_path = base_path / "data_for_tables.xlsx"
    try:
        data = pd.read_excel(file_path, sheet_name="validation scores")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    r2_data = data.loc[:, ["r2", "as7262", "as7263", "as7265x"]]
    mae_data = data.loc[:, ["mae", "as7262.1", "as7263.1", "as7265x.1"]]

    # Rename columns for clarity
    r2_data.columns = ["Leaf", "AS7262", "AS7263", "AS7265x"]
    mae_data.columns = ["Leaf", "AS7262", "AS7263", "AS7265x"]

    # Set the index to "Crop" for heatmaps
    r2_data.set_index("Leaf", inplace=True)
    mae_data.set_index("Leaf", inplace=True)
    # print(r2_data)
    for df in [r2_data, mae_data]:
        df.index = [leaf.capitalize() for leaf in df.index]

    # Create heatmaps
    plot_heatmaps(r2_data, mae_data, filename=filename)


def make_table(sensors: list[str], filename: str = None, random_state=None) -> None:
    """
    Evaluates regression models for a list of sensors across multiple leaves,
    calculates R² and MAE scores, and generates heatmaps of the results.

    Parameters
    ----------
    sensors : list of str
        A list of sensor names to be evaluated (e.g., "as7262", "as7263").
    filename : str, optional
        The name of the file to save the generated heatmaps as a PDF. If
        a falsy value (e.g., `None`), the heatmaps will not be saved (default is `None`).
    random_state: int, optional
        The random state to pass to the helper_functions.evaluate_model_scores function.
        Used so when making figures, the numbers will stay the same so the manuscript does not
        have to be updated if making small changes to figure.
        Defaults to no random_state which will make each new run slightly different.

    Returns
    -------
    None
        Displays R² and MAE pivot tables and generates heatmaps.
        Optionally saves the heatmaps if `filename` is specified.

    Notes
    -----
    - For each sensor and leaf combination, the function evaluates a regression model using
      partial least squares regression (PLS) with specific configurations
      based on the sensor type.
    - Data for each sensor and leaf is preprocessed and scaled before model evaluation.
    - The results are stored in a DataFrame, which is then pivoted to create tables
      for R² and MAE scores.
    - Heatmaps for R² and MAE are generated and displayed with the function plot_heatmaps.
    """
    # Initialize an empty list to store results
    results = []

    # Loop over each sensor and leaf combination
    for sensor in sensors:
        for leaf in LEAVES:
            print(sensor, leaf)
            # set the proper led for the sensor
            # led = "White LED"

            if sensor == "as7262":
                regressor = PLSRegression(n_components=pls_n_comps[sensor][leaf])
            elif sensor == "as7263":
                regressor = PLSRegression(n_components=pls_n_comps[sensor][leaf])
            elif sensor == "as7265x":
                # led = "b'White IR'"
                regressor = PLSRegression(n_components=pls_n_comps[sensor][leaf])
                # regressor = LassoLarsIC("aic")
                # regressor = ARDRegression(lambda_2=0.001)
            elif sensor in ["as72651", "as72652", "as72653"]:
                regressor = PLSRegression(n_components=5)

            # Get the data for the current sensor and leaf
            # x, y, groups = get_data.get_x_y(
            #     sensor=sensor,
            #     leaf=leaf,
            #     measurement_type="absorbance",
            #     led=led,
            #     int_time=50,
            #     led_current="12.5 mA",
            #     send_leaf_numbers=True
            # )
            # x_fluro = None
            # if sensor == "as7265x":
            #     x_fluro, _, _ = get_data.get_x_y(
            #             sensor=sensor,
            #             leaf=leaf,
            #             measurement_type="raw",
            #             led="b'UV'",
            #             int_time=150,
            #             led_current="12.5 mA",
            #             send_leaf_numbers=True
            #         )
            #     x, y, groups = get_data.get_x_y(
            #         sensor=sensor,
            #         leaf=leaf,
            #         measurement_type="absorbance",
            #         led="b'White IR'",
            #         int_time=50,
            #         led_current="12.5 mA",
            #         send_leaf_numbers=True
            #     )
            if sensor in SENSORS:
                x, y, groups = get_data.get_cleaned_data(sensor, leaf,
                                                         mean=False)
            else:
                x, y, groups = get_data.get_cleaned_data("as7265x", leaf,
                                                         mean=False)
            if sensor == "as72651":
                x = x[["610 nm", "680 nm", "730 nm", "760 nm", "810 nm", "860 nm"]]
            elif sensor == "as72652":
                x = x[["560 nm", "585 nm", "645 nm", "705 nm", "900 nm", "940 nm"]]
            elif sensor == "as72653":
                x = x[["410 nm", "435 nm", "460 nm", "485 nm", "510 nm", "535 nm"]]

            x = StandardScaler().fit_transform(x)
            y = y['Avg Total Chlorophyll (µg/cm2)']

            # Convert y to a pandas Series with groups as the index,
            # they need to have the same index
            y = pd.DataFrame({'y': y, 'group': groups})
            # reset the group to the y index
            y.set_index('group', inplace=True)
            x = np.array(x)
            # x = StandardScaler().fit_transform(x)
            # x['fluro1'] = x_fluro["680 nm"] - x_fluro["610 nm"]
            # x['fluro2'] = x_fluro["730 nm"] - x_fluro["610 nm"]
            # x['fluro3'] = x_fluro["680 nm"] / x_fluro["730 nm"]
            # print(x)
            # x = np.array(x)
            # Evaluate the scores for the current data
            scores = helper_functions.evaluate_model_scores(
                x, y, groups, regressor=regressor, n_splits=N_SPLITS,
                group_by_mean=False, random_state=random_state
            )

            results.append({
                'Sensor': sensor,
                'Leaf': leaf,
                "R2": scores['test_r2'][0],
                "R2 std": scores['test_r2'][1],
                "MAE": scores['test_mae'][0],
                "MAE std": scores['test_mae'][1]
            })
    results_df = pd.DataFrame(results)
    print(results_df)
    # Pivot the table so that each sensor has its own column
    pivot_df_r2 = results_df.pivot(index='Leaf', columns='Sensor', values='R2')
    pivot_df_mae = results_df.pivot(index='Leaf', columns='Sensor', values='MAE')

    # Display the pivot tables for R2 and MAE
    print("R² Scores Table:")
    print(pivot_df_r2)

    print("\nMAE Scores Table:")
    print(pivot_df_mae)

    # plot_grouped_bar_charts(results_df)
    r2_pivot = results_df.pivot(index='Leaf', columns='Sensor', values='R2')
    mae_pivot = results_df.pivot(index='Leaf', columns='Sensor', values='MAE')
    for df in [r2_pivot, mae_pivot]:
        # Capitalize "AS" in the column names
        df.columns = [col.replace('as', 'AS') for col in df.columns]

        # Capitalize leaf names in the index
        df.index = [leaf.capitalize() for leaf in df.index]
    plot_heatmaps(r2_pivot, mae_pivot, filename=filename)


def plot_heatmaps(r2_data: pd.DataFrame, mae_data: pd.DataFrame,
                  filename: str = "", axes=None) -> None:
    """
    Plots heatmaps for R² and MAE. If axes is provided, the heatmaps are plotted on the given axes.
    Otherwise, subplots are created automatically.

    Parameters
    ----------
    r2_data : pd.DataFrame
        R² values for the heatmap.
    mae_data : pd.DataFrame
        MAE values for the heatmap.
    filename : str, optional
        The name of the file to save the heatmaps as a PDF. If a falsy value
        (e.g., "" or None) is provided, the heatmaps are not saved (default is "").
    axes : array-like, optional
        The axes on which to plot the heatmaps. If None, subplots are created.

    Returns
    -------
    None
        Displays the heatmaps and optionally saves them to a file.
    """
    # If axes is None, create subplots
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # R2 Heatmap
    sns.heatmap(r2_data, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={"label": "$R^2$ Score"}, ax=axes[0])
    axes[0].set_title("Validation $R^2$ Scores")

    # MAE Heatmap
    sns.heatmap(mae_data, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "Mean Absolute Error\n(MAE) (µg/cm2)"},
                ax=axes[1])
    axes[1].set_title("Validation MAE Scores")
    # axes[1].set_xlabel("Sensor")
    axes[0].set_ylabel("Leaf Species", fontsize=12)
    axes[1].set_ylabel("")
    for i in [0, 1]:
        axes[i].set_xlabel("Sensors")
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=45)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].annotate(f"{chr(i + 97)})", (-0.4, 1.1), xycoords='axes fraction',
                         fontsize=12, fontweight='bold', va='top')

    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust(left=0.15, wspace=0.52, bottom=0.2)
    if filename:
        fig.savefig(filename, dpi=600)
    plt.show()


def plot_grouped_bar_charts(data):
    """
    Plot two grouped bar charts side-by-side for R2 and MAE values
    for each leaf and sensor combination.

    Parameters:
        data (pd.DataFrame): A DataFrame containing the following columns:
            - 'Sensor': The sensor type.
            - 'Leaf': The type of leaf.
            - 'R2': Mean R2 scores.
            - 'R2 std': Standard deviation of the R2 scores.
            - 'MAE': Mean MAE scores.
            - 'MAE std': Standard deviation of the MAE scores.
    """
    # Get unique leaves and sensors
    leaves = data['Leaf'].unique()
    sensors = data['Sensor'].unique()

    # Set bar width and positions
    bar_width = 0.25
    num_leaves = len(leaves)
    br_positions = [np.arange(num_leaves)]  # Base positions for the bars

    # Create positions for each sensor's bars
    for i in range(1, len(sensors)):
        br_positions.append([x + bar_width for x in br_positions[i - 1]])

    # Initialize the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Plot R2 bars
    for idx, sensor in enumerate(sensors):
        sensor_data = data[data['Sensor'] == sensor]
        r2_means = sensor_data['R2']
        r2_stds = sensor_data['R2 std']

        ax1.bar(br_positions[idx], r2_means, yerr=r2_stds, width=bar_width,
                edgecolor='grey', label=sensor, capsize=5)

    # Customize the first plot (R2)
    ax1.set_xlabel('Leaf Type', fontweight='bold', fontsize=12)
    ax1.set_ylabel('R2 Score', fontweight='bold', fontsize=12)
    ax1.set_xticks([r + bar_width for r in range(len(leaves))])
    ax1.set_xticklabels(leaves)
    ax1.set_title('R2 Scores by Leaf and Sensor', fontweight='bold', fontsize=14)
    ax1.set_ylim([.4, 1.0])

    # Plot MAE bars
    for idx, sensor in enumerate(sensors):
        sensor_data = data[data['Sensor'] == sensor]
        mae_means = sensor_data['MAE']
        mae_stds = sensor_data['MAE std']

        ax2.bar(br_positions[idx], mae_means, yerr=mae_stds, width=bar_width,
                edgecolor='grey', label=sensor, capsize=5)

    # Customize the second plot (MAE)
    ax2.set_xlabel('Leaf Type', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MAE Score', fontweight='bold', fontsize=12)
    ax2.set_xticks([r + bar_width for r in range(len(leaves))])
    ax2.set_xticklabels(leaves)
    ax2.set_title('MAE Scores by Leaf and Sensor', fontweight='bold', fontsize=14)
    ax2.legend(title='Sensor', loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ["as7265x", "as72651", "as72652", "as72653"]
    # make_table(["as7262", "as7263", "as7265x"],
    #            filename="3_sensors.pdf")
    # make_table(["as7262"])
    #            # filename="Individual_AS7265x_chips.pdf")
    make_table(["as7265x", "as72651", "as72652", "as72653"],
               filename="as7265x_individual_sensors.jpeg",
               random_state=56)

    # create_validation_heatmaps(filename="validation_scores.jpeg")
