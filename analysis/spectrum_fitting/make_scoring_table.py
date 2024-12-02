# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data
import helper_functions
import preprocessors
SENSORS = ["as7262", "as7263", "as7265x"]
LEAVES = ["banana", "jasmine", "mango", "rice", "sugarcane"]


def make_table():
    # Initialize an empty list to store results
    results = []

    # Loop over each sensor and leaf combination
    for sensor in SENSORS:
        for leaf in LEAVES:
            # set the proper led for the sensor
            led = "White LED"

            if sensor == "as7262":
                regressor = PLSRegression(n_components=6)
            elif sensor == "as7263":
                regressor = PLSRegression(n_components=5)
            elif sensor == "as7265x":
                led = "b'White IR'"
                regressor = PLSRegression(n_components=16)

            # regressor = TransformedTargetRegressor(
            #     regressor=regressor, func=neg_log, inverse_func=neg_exp)

            # Get the data for the current sensor and leaf
            x, y, groups = get_data.get_x_y(
                sensor=sensor,
                leaf=leaf,
                measurement_type="reflectance",
                led=led,
                int_time=50,
                led_current="12.5 mA",
                send_leaf_numbers=True
            )
            # x = 1/x
            # # process the data
            # if sensor != "as7265x":
            #     x = PolynomialFeatures(degree=2, include_bias=False
            #                            ).fit_transform(x)
            y = y['Avg Total Chlorophyll (µg/cm2)']
            # Convert y to a pandas Series with groups as the index,
            # they need to have the same index
            y = pd.DataFrame({'y': y, 'group': groups})
            # reset the group to the y index
            y.set_index('group', inplace=True)

            x = StandardScaler().fit_transform(x)
            # x = np.array(x)
            # Evaluate the scores for the current data
            scores = helper_functions.evaluate_model_scores(
                x, y, groups, regressor=regressor, n_splits=100,
                group_by_mean=True
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
    plot_heatmaps(results_df)

    # Plot the R² scores as a heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(pivot_df_r2, annot=True, fmt=".2f", cmap="viridis")
    # plt.title("R² Scores by Leaf and Sensor")
    # plt.ylabel("Leaf")
    # plt.xlabel("Sensor")
    # plt.show()
    #
    # # Plot the MAE scores as a heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(pivot_df_mae, annot=True, fmt=".2f", cmap="viridis")
    # plt.title("MAE Scores by Leaf and Sensor")
    # plt.ylabel("Leaf")
    # plt.xlabel("Sensor")
    # plt.show()


def plot_heatmaps(data, r2_col='R2', mae_col='MAE', sensor_col='Sensor', leaf_col='Leaf'):
    """
    Plots two heatmaps side by side: one for R2 and one for MAE.

    Args:
        data (pd.DataFrame): DataFrame containing the data to plot.
        r2_col (str): Column name for R2 values.
        mae_col (str): Column name for MAE values.
        sensor_col (str): Column name for sensors.
        leaf_col (str): Column name for leaves.

    Returns:
        None: Displays the plots.
    """
    # Pivot data for heatmaps
    r2_pivot = data.pivot(index=leaf_col, columns=sensor_col, values=r2_col)
    mae_pivot = data.pivot(index=leaf_col, columns=sensor_col, values=mae_col)

    # Create subplots for the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    # R2 Heatmap
    sns.heatmap(r2_pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title("R2 Heatmap")
    axes[0].set_ylabel("Leaf")
    axes[0].set_xlabel("Sensor")

    # MAE Heatmap
    sns.heatmap(mae_pivot, annot=True, fmt=".2f", cmap="viridis", ax=axes[1])
    axes[1].set_title("MAE Heatmap")
    axes[1].set_xlabel("Sensor")

    # Adjust layout
    plt.tight_layout()
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
    make_table()
