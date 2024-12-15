# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

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
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data
import helper_functions
import preprocessors
SENSORS = ["as7262", "as7263", "as7265x"]
LEAVES = ["banana", "jasmine", "mango", "rice", "sugarcane"]
pd.set_option('display.max_rows', 500)
N_SPLITS = 10  # set to 400 for low variance, 10 to be quick


def create_validation_heatmaps():
    """
    Load data from an Excel file and create heatmaps for R2 and MAE.

    Parameters:
        excel_file (str): Path to the Excel file.
        sheet_name (str): Name of the worksheet to load.

    Returns:
        None: Displays the heatmaps.
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
    print(data)
    r2_data = data.loc[:, ["r2", "as7262", "as7263", "as7265x"]]
    mae_data = data.loc[:, ["mae", "as7262.1", "as7263.1", "as7265x.1"]]

    # Rename columns for clarity
    r2_data.columns = ["Leaf", "AS7262", "AS7263", "AS7265x"]
    mae_data.columns = ["Leaf", "AS7262", "AS7263", "AS7265x"]

    # Set the index to "Crop" for heatmaps
    r2_data.set_index("Leaf", inplace=True)
    mae_data.set_index("Leaf", inplace=True)
    print(r2_data)
    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    sns.heatmap(
        r2_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "$R^2$"},
        ax=axes[0]
    )
    axes[0].set_title("Validation $R^2$ Scores")

    sns.heatmap(
        mae_data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Mean Absolute Error\n(MAE) (µg/cm2)"},
        ax=axes[1]
    )
    axes[1].set_title("Validation MAE Scores")
    axes[1].set_ylabel("")
    for i in [0, 1]:
        axes[i].set_xlabel("Sensors")
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=45)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].annotate(f"{chr(i + 97)})", (-0.4, 1.1), xycoords='axes fraction',
                         fontsize=12, fontweight='bold', va='top')

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.52, bottom=0.2)
    # fig.savefig("validation_scores.pdf", format='pdf')
    plt.show()


def make_table(sensors: list[str], use_fluro=False):
    # Initialize an empty list to store results
    results = []

    # Loop over each sensor and leaf combination
    for sensor in sensors:
        for leaf in LEAVES:
            print(sensor, leaf)
            # set the proper led for the sensor
            led = "White LED"

            if sensor == "as7262":
                regressor = PLSRegression(n_components=5)
            elif sensor == "as7263":
                regressor = PLSRegression(n_components=5)
            elif sensor == "as7265x":
                led = "b'White IR'"
                regressor = PLSRegression(n_components=10)
                # regressor = LassoLarsIC("aic")
                # regressor = ARDRegression(lambda_2=0.001)
            elif sensor in ["as72651", "as72652", "as72653"]:
                regressor = PLSRegression(n_components=5)

            # regressor = TransformedTargetRegressor(
            #     regressor=regressor, func=neg_log, inverse_func=neg_exp)

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
            # x = PolynomialFeatures(degree=2).fit_transform(x)
            y = y['Avg Total Chlorophyll (µg/cm2)']

            # x = x.reset_index(drop=True)
            # x_fluro = x_fluro.reset_index(drop=True)
            # x['fluro1'] = x_fluro["680 nm"] - x_fluro["610 nm"]
            # x['fluro2'] = x_fluro["730 nm"] - x_fluro["610 nm"]
            # x['fluro3'] = x_fluro["680 nm"] / x_fluro["730 nm"]
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
                group_by_mean=False
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

    plot_heatmaps(r2_pivot, mae_pivot)


def plot_heatmaps(r2_data: pd.DataFrame, mae_data: pd.DataFrame, filename: str="") -> None:
    """
    Plots two heatmaps side by side: one for R² and one for MAE.

    Parameters
    ----------
    r2_data : pd.DataFrame
        DataFrame containing the R² values for the heatmap.
    mae_data : pd.DataFrame
        DataFrame containing the MAE values for the heatmap.
    filename : str, optional
        The name of the file to save the heatmaps as a PDF. If a falsy value
        (e.g., "" or None) is provided, the heatmaps are not saved (default is "").

    Returns
    -------
    None
        Displays the heatmaps and optionally saves them to a file.
    """
    # Create subplots for the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # R2 Heatmap
    sns.heatmap(r2_data, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={"label": "$R^2$"}, ax=axes[0])
    axes[0].set_title("Validation $R^2$ Scores")

    # MAE Heatmap
    sns.heatmap(mae_data, annot=True, fmt=".2f", cmap="viridis",
                cbar_kws={"label": "Mean Absolute Error\n(MAE) (µg/cm2)"},
                ax=axes[1])
    axes[1].set_title("Validation MAE Scores")
    # axes[1].set_xlabel("Sensor")
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
        fig.savefig("validation_scores.pdf", format='pdf')
    plt.show()


def plot_grouped_bar_charts(data):
    """
    Plot two grouped bar charts side-by-side for R2 and MAE values for each leaf and sensor combination.

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
    barWidth = 0.25
    num_leaves = len(leaves)
    br_positions = [np.arange(num_leaves)]  # Base positions for the bars

    # Create positions for each sensor's bars
    for i in range(1, len(sensors)):
        br_positions.append([x + barWidth for x in br_positions[i - 1]])

    # Initialize the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Plot R2 bars
    for idx, sensor in enumerate(sensors):
        sensor_data = data[data['Sensor'] == sensor]
        r2_means = sensor_data['R2']
        r2_stds = sensor_data['R2 std']

        ax1.bar(br_positions[idx], r2_means, yerr=r2_stds, width=barWidth,
                edgecolor='grey', label=sensor, capsize=5)

    # Customize the first plot (R2)
    ax1.set_xlabel('Leaf Type', fontweight='bold', fontsize=12)
    ax1.set_ylabel('R2 Score', fontweight='bold', fontsize=12)
    ax1.set_xticks([r + barWidth for r in range(len(leaves))])
    ax1.set_xticklabels(leaves)
    ax1.set_title('R2 Scores by Leaf and Sensor', fontweight='bold', fontsize=14)
    ax1.set_ylim([.4, 1.0])

    # Plot MAE bars
    for idx, sensor in enumerate(sensors):
        sensor_data = data[data['Sensor'] == sensor]
        mae_means = sensor_data['MAE']
        mae_stds = sensor_data['MAE std']

        ax2.bar(br_positions[idx], mae_means, yerr=mae_stds, width=barWidth,
                edgecolor='grey', label=sensor, capsize=5)

    # Customize the second plot (MAE)
    ax2.set_xlabel('Leaf Type', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MAE Score', fontweight='bold', fontsize=12)
    ax2.set_xticks([r + barWidth for r in range(len(leaves))])
    ax2.set_xticklabels(leaves)
    ax2.set_title('MAE Scores by Leaf and Sensor', fontweight='bold', fontsize=14)
    ax2.legend(title='Sensor', loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    make_table(["as7265x", "as72651", "as72652", "as72653"])
    # create_validation_heatmaps()
