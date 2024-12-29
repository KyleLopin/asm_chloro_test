# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Initial functions developed to fit the spectrum from the as7262, as7263 and as7265x sensor
that measured leaf spectrum to calculate their chlorophyll.

Not all functions are working
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from cycler import cycler

# installed libraries
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin  # for type-hinting
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import (cross_val_predict, LearningCurveDisplay,
                                     LeaveOneOut)
from sklearn.preprocessing import StandardScaler

# local files
import get_data
import helper_functions
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# plt.style.use('ggplot')
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams['axes.prop_cycle'] = cycler(color=default_colors)

pls_n_comps = {"as7262": {"banana": 6, "jasmine": 4, "mango": 5, "rice": 6, "sugarcane": 6},
               "as7263": {"banana": 5, "jasmine": 5, "mango": 5, "rice": 5, "sugarcane": 5},
               "as7265x": {"banana": 8, "jasmine": 14, "mango": 11, "rice": 6, "sugarcane": 6}}

CV_REPEATS = 10  # 200 for figure, 10 for formatting figure
TEST_SIZE = 0.2
CV = StratifiedGroupShuffleSplit(n_splits=CV_REPEATS, test_size=TEST_SIZE, n_bins=10)
SENSORS = ["as7262", "as7263", "as7265x"]
TRAINING_SIZES = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Plot formatting options
TEST_COLOR = 'forestgreen'
TRAIN_COLOR = 'indigo'
MARKER_SIZE = 15
MAE_LIM = [0, -8]
LEFT_ALIGN = 0.03
RIGHT_ALIGN = 0.8


def graph_y_predicted_with_loo(
        x=None, y=None, groups=None,
        regressor: RegressorMixin = LinearRegression,
        y_column: str = "Avg Total Chlorophyll (µg/cm2)"):
    """
    Graph the predicted chlorophyll versus actual chlorophyll levels using
    Leave-One-Out cross-validation.

    Args:
        x (array-like): Feature data for training.
        y (array-like): Target data.
        groups (array-like): Group labels corresponding to each data point.
        regressor (RegressorMixin, optional): The regression model to use.
        y_column (str): The label for the y-axis in the graph.
    """
    # Ensure that y is a pd.Series with the index as the group
    df = pd.DataFrame({'y': y, 'group': groups})
    df.set_index('group', inplace=True)

    # Prepare the Leave-One-Out cross-validator
    loo = LeaveOneOut()

    # Use cross_val_predict to get the predicted values for each measurement
    y_pred = cross_val_predict(regressor, x, y, cv=loo)

    # Create a DataFrame for predictions and actual values
    predictions_df = pd.DataFrame({'y_true': df['y'], 'y_pred': y_pred}, index=df.index)

    # Group by the original index (group) and calculate mean predictions and true values
    grouped_predictions = predictions_df.groupby(predictions_df.index).mean()

    # Calculate R^2 score and Mean Absolute Error
    r2 = r2_score(grouped_predictions['y_true'], grouped_predictions['y_pred'])
    mae = mean_absolute_error(grouped_predictions['y_true'], grouped_predictions['y_pred'])

    # Print the scores
    print(f'R^2 Score: {r2:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(grouped_predictions['y_true'], grouped_predictions['y_pred'], color='blue',
                label='Predicted vs Actual')
    plt.plot([grouped_predictions['y_true'].min(), grouped_predictions['y_true'].max()],
             [grouped_predictions['y_true'].min(), grouped_predictions['y_true'].max()],
             color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel(y_column)
    plt.ylabel('Predicted')
    plt.title('Leave-One-Out Cross-Validation: Predicted vs Actual')
    plt.legend()
    plt.grid()
    plt.show()


def graph_y_predicted_vs_y_actual(
        x: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        regressor=None,
        ax=None):
    """
    Graph the predicted chlorophyll versus actual chlorophyll levels.

    Args:
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target values.
        groups (pd.Series): Group labels corresponding to each data point.
        regressor: The regression model (if None, defaults to LinearRegression).
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
    """
    # Convert y to a pandas Series with groups as the index,
    # they need to have the same index
    y = pd.DataFrame({'y': y, 'group': groups})
    # reset the group to the y index
    y.set_index('group', inplace=True)

    if ax is None:
        fig, ax = plt.subplots()

    # Split the data into training and testing sets
    splitter = CV
    train_idx, test_idx = next(splitter.split(x, y, groups))
    print("after splitter")
    print(y.shape)
    print(y.iloc[train_idx].shape)
    # Train and test data
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit the regressor on the training data
    regressor.fit(x_train, y_train)

    # Make predictions on both training and testing data
    y_train_pred = regressor.predict(x_train)
    y_test_pred = regressor.predict(x_test)
    print("y train shapes")
    print(y_train_pred.shape)
    print(y_train.shape)

    # Convert predictions to pandas Series for grouping,
    # use flatten to convert (270, 1) shape to (270,)
    y_train_pred = pd.Series(y_train_pred.flatten(), index=y_train.index)
    y_test_pred = pd.Series(y_test_pred.flatten(), index=y_test.index)

    # Group by the group index and calculate the mean
    y_train_grouped = y_train.groupby(y_train.index).mean()
    y_test_grouped = y_test.groupby(y_test.index).mean()
    y_train_pred_grouped = y_train_pred.groupby(y_train_pred.index).mean()
    y_test_pred_grouped = y_test_pred.groupby(y_test_pred.index).mean()

    # get general error measurements
    # scores = helper_functions.evaluate_model_scores(
    #     x, y, groups, regressor=regressor, n_splits=20)
    # print(scores)
    print('===== making scores')
    scores = helper_functions.evaluate_model_scores(
        x, y, groups, regressor=regressor, n_splits=CV_REPEATS,
        test_size=TEST_SIZE, group_by_mean=True)
    print(scores)

    # Plot the results
    ax.scatter(y_train_grouped, y_train_pred_grouped, label='Training set',
               color=TRAIN_COLOR, alpha=0.6)
    ax.scatter(y_test_grouped, y_test_pred_grouped, label='Test set',
               color=TEST_COLOR, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    # plt.xlabel("Measured Chlorophyll (µg/cm²)")

    ax.annotate(f"R² test  ={scores['test_r2'][0]:.2f}", xy=(LEFT_ALIGN, 0.60),
                xycoords='axes fraction', color='#101028', fontsize='large')
    ax.annotate(f"R² train ={scores['train_r2'][0]:.2f}", xy=(LEFT_ALIGN, 0.44),
                xycoords='axes fraction', color='#101028', fontsize='large')
    ax.annotate(f"MAE test  ={scores['test_mae'][0]:.2f}", xy=(RIGHT_ALIGN, 0.30),
                xycoords='axes fraction', color='#101028', fontsize='large')
    ax.annotate(f"MAE train ={scores['train_mae'][0]:.2f}", xy=(RIGHT_ALIGN, 0.14),
                xycoords='axes fraction', color='#101028', fontsize='large')

    # plt.legend(loc='lower right')
    plt.grid(True)
    if ax is None:
        plt.show()


def make_leaf_figure(leaf: str):
    # learning_curves.set_style()  # keep orange and blue colors like classic style
    # Create the figure and GridSpec
    figure = plt.figure(figsize=(7, 8), constrained_layout=False)
    grid = GridSpec(4, 3, figure=figure)  # 4 rows, 3 columns

    # Top 3 full-width axes
    pred_axes = [
        figure.add_subplot(grid[0, :]),  # Row 0, all columns
        figure.add_subplot(grid[1, :]),  # Row 1, all columns
        figure.add_subplot(grid[2, :])  # Row 2, all columns
    ]

    # Bottom 3 separate axes
    lc_axes = [
        figure.add_subplot(grid[3, 0]),  # Row 3, column 0
        figure.add_subplot(grid[3, 1]),  # Row 3, column 1
        figure.add_subplot(grid[3, 2])  # Row 3, column 2
    ]
    figure.suptitle(f"{leaf.capitalize()} Leaf Total Chlorophyll", fontsize=14)
    for i, sensor in enumerate(SENSORS):
        x, y, groups = get_data.get_cleaned_data(sensor, leaf)

        pls = PLSRegression(n_components=pls_n_comps[sensor][leaf])

        y = y['Avg Total Chlorophyll (µg/cm2)']
        x = StandardScaler().fit_transform(x)

        graph_y_predicted_vs_y_actual(x, y, groups, regressor=pls,
                                      ax=pred_axes[i])
        LearningCurveDisplay.from_estimator(
            pls, x, y, groups=groups, cv=CV, scoring='neg_mean_absolute_error',
            train_sizes=TRAINING_SIZES,
            std_display_style="fill_between", ax=lc_axes[i])
        # uniform set y axis
        lc_axes[i].set_ylim(MAE_LIM)
        # Update x-axis ticks and labels
        # convert TRAINING_SIZES to number of leaves
        num_leaves = [x * 300 * (1 - TEST_SIZE) for x in TRAINING_SIZES]
        # Calculate the labels, skipping half of them
        labels = [int(size / 3) if idx % 2 == 0 else "" for idx, size in enumerate(num_leaves)]

        lc_axes[i].set_xticks(ticks=num_leaves,
                              labels=labels)
        lc_axes[i].set_xlabel("Number of Leaves in Training Set")  # Update x-axis label
        pred_axes[i].annotate(f"AS{sensor[2:]}", xy=(LEFT_ALIGN, 0.83),
                              xycoords='axes fraction', color='#101028', fontsize='large',
                              fontweight='bold')
        lc_axes[i].annotate(f"AS{sensor[2:]}", xy=(0.4, 0.12),
                            xycoords='axes fraction', color='#101028', fontsize='large')
        if i == 0:
            pred_axes[i].legend(loc='upper center', bbox_to_anchor=(0.45, 1.05),
                                ncol=2, frameon=True)
            lc_axes[i].set_yticks([0, -2, -4, -6, -8])  # Set the tick positions
            # Set the corresponding labels as positive
            lc_axes[i].set_yticklabels([0, 2, 4, 6, 8])
            lc_axes[i].set_ylabel(r"Mean Absolute Error\n($\mu$g/cm$^2$)")

        if i != 0:
            lc_axes[i].set_ylabel("")
            lc_axes[i].set_yticklabels([])  # Turn off xtick labels
        if i != 1:
            lc_axes[i].set_xlabel("")
        if i != 2:
            lc_axes[i].legend().remove()
    pred_axes[2].set_xlabel("Predicted Chlorophyll (µg/cm²)")

    # Adjust the layout to make room for the Y-axis label and the sides
    # Do not use .tight_layout
    figure.subplots_adjust(left=0.11, right=0.98,
                           top=0.94, bottom=0.07, hspace=0.4)

    # Add a single Y-axis label for all subplots
    figure.text(0.04, 0.6, r'Measured Chlorophyll ($\mu$g/cm$^2$)', va='center',
                rotation='vertical',
                fontsize=12)

    # Save the figure
    # figure.savefig(f'{leaf}_leaf_total_chlorophyll.jpg', dpi=600)
    plt.show()


if __name__ == '__main__':
    make_leaf_figure("jasmine")
