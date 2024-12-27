# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from cycler import cycler

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit, LearningCurveDisplay
from sklearn.preprocessing import (StandardScaler)
# plt.style.use("ggplot")

# local files
import get_data
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit


YLIM = ([0, -8])
TRAINING_SIZES = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TEST_SIZE = 0.2
SCORING = "r2"
SCORING = "neg_mean_absolute_error"
LINESTYLES = ['-', '--', ':', '-', '-', ':', '-', '-', ':']
COLORS = ['black', 'blue', 'red', "green", 'magenta', 'cyan', 'darkred',
          'cyan', 'gold']
LEAF = "banana"
SENSORS = ["as7262", "as7263", "as7265x"]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]

pls_n_comps = {"as7262": {"banana": 6, "jasmine": 4, "mango": 5, "rice": 6, "sugarcane": 6},
               "as7263": {"banana": 5, "jasmine": 5, "mango": 5, "rice": 5, "sugarcane": 5},
               "as7265x": {"banana": 8, "jasmine": 14, "mango": 11, "rice": 6, "sugarcane": 6}}

CV = GroupShuffleSplit(test_size=TEST_SIZE, n_splits=10)
N_SPLITS = 200  # change to make faster
CV = StratifiedGroupShuffleSplit(test_size=TEST_SIZE, n_splits=N_SPLITS,
                                 n_bins=10, random_state=128)
# CV = ShuffleSplit(test_size=TEST_SIZE, n_splits=N_SPLITS)


def set_style():
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.style.use('ggplot')
    plt.rcParams['axes.prop_cycle'] = cycler(color=default_colors)


def make_learning_curves(ax: plt.Axes, x: pd.DataFrame,
                         y: pd.Series, groups: pd.Series, i: int,
                         name: str,
                         regr: BaseEstimator,
                         score: str = SCORING):
    LearningCurveDisplay.from_estimator(regr, x, y, groups=groups,
                                        cv=CV, scoring=score, n_jobs=-1,
                                        train_sizes=TRAINING_SIZES,
                                        ax=ax, std_display_style=None,
                                        line_kw={"color": COLORS[i],
                                                 "linestyle": LINESTYLES[i]})
    ax.plot([], [], label=name, color=COLORS[i], linestyle=LINESTYLES[i])


def make_all_leaf_learning_curves():
    set_style()
    figure, axes = plt.subplots(5, 3, figsize=(7, 8.5),
                                sharex=True, sharey=True)

    for row, leaf in enumerate(["banana", "jasmine", "mango", "rice", "sugarcane"]):
        for column, sensor in enumerate(SENSORS):
            print(leaf, sensor, row, column)
            # get the data
            x, y, groups = get_data.get_cleaned_data(sensor, leaf,
                                                     mean=False)
            y = y['Avg Total Chlorophyll (µg/cm2)']
            x = StandardScaler().fit_transform(x)
            # get the number of components optimized for sensor / leaf combination
            n_components = pls_n_comps[sensor][leaf]
            regressor = PLSRegression(n_components=n_components)
            LearningCurveDisplay.from_estimator(
                regressor, x, y, groups=groups, cv=CV, scoring=SCORING,
                n_jobs=-1, train_sizes=TRAINING_SIZES,
                ax=axes[row][column])
            axes[row][column].set_xlabel("")
            axes[row][column].set_ylabel("")
            if row != 2 or column != 1:
                axes[row][column].legend().remove()
            if row == 0:
                axes[row][column].set_title("AS" + sensor[2:], fontsize=10)

            # if SCORING == 'neg_mean_absolute_error':
                # display.train_scores = -display.train_scores
                # display.test_scores = -display.test_scores
                # Plot with adjusted values
                # display.plot(ax=axes[row][column])
                # axes[row][column].plot(-display.train_scores, -display.test_scores)
            if row == 4:
                # convert TRAINING_SIZES to number of leaves
                num_leaves = [x * 300 * (1 - TEST_SIZE) for x in TRAINING_SIZES]
                # Calculate the labels, skipping half of them
                labels = [int(size / 3) if idx % 2 == 0
                          else "" for idx, size in enumerate(num_leaves)]

                axes[row][column].set_xticks(ticks=num_leaves,
                                             labels=labels)
                if column == 1:  # Update x-axis label
                    axes[row][column].set_xlabel("Number of Leaves in Training Set")


            if row == 2 and column == 0:
                axes[row][column].set_ylabel(
                    "Mean Absolute Error of Total Chlorophyll ($\mu$g/cm$^2$)")
            # convert the negative mean absolute to positive by reversing the y-axis
            axes[row][column].set_ylim(YLIM)

        # convert the negative mean absolute to positive
        axes[row][0].set_yticks([0, -2, -4, -6], [0, 2, 4, 6])

        axes[row][0].annotate(f"{leaf.capitalize()}", (0.5, 0.1),
                              xycoords='axes fraction', fontsize=10)
    # add annotations for the panels
    i = 0
    for column, sensor in enumerate(SENSORS):
        for row, leaf in enumerate(["banana", "jasmine", "mango", "rice", "sugarcane"]):
            axes[row][column].annotate(f"{chr(i + 97)})", (0.10, 0.05),
                                       fontsize=10, fontweight='bold',
                                       xycoords='axes fraction')
            i += 1

    figure.suptitle("Validation curves", fontsize=14)
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.97, top=0.92, wspace=0.08)
    figure.savefig("MEA_Validation_Curves.png", dpi=600)
    plt.show()


def make_learning_curves_3_sensors(leaf: str):
    """
    Generate learning curve plots for three measurement modes of a specified leaf and sensor.
    Used to compare which measurement mode is best to fit the model

    Parameters
    ----------
    leaf : str
        Name of the leaf type to analyze (e.g., 'mango', 'banana').

    Returns
    -------
    None
        This function generates and displays a figure containing three subplots,
        each representing a learning curve for a different measurement mode.
    """

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.style.use('ggplot')
    plt.rcParams['axes.prop_cycle'] = cycler(color=default_colors)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3),
                             sharey=True)
    fig.suptitle(f"{leaf.capitalize()} Leaf Validation Curves", y=0.95)
    for i, sensor in enumerate(SENSORS):
        print(f'sensor: {sensor}')
        x, y, groups = get_data.get_cleaned_data(sensor, leaf,
                                                 mean=False)
        y = y['Avg Total Chlorophyll (µg/cm2)']
        x = StandardScaler().fit_transform(x)

        n_components = pls_n_comps[sensor][leaf]
        regressor = PLSRegression(n_components=n_components)

        x = np.array(x, copy=True)
        y = np.array(y, copy=True)
        LearningCurveDisplay.from_estimator(
            regressor, x, y, groups=groups, cv=CV, scoring=SCORING,
            train_sizes=TRAINING_SIZES,
            std_display_style="fill_between", ax=axes[i])
        if i < 2:
            axes[i].get_legend().remove()
        if i > 0:
            axes[i].set_ylabel("")  # Remove the ylabel
        else:
            axes[i].set_ylabel("$R^2$")
        axes[i].set_ylim(YLIM)

        # Update x-axis ticks and labels
        # convert TRAINING_SIZES to number of leaves
        num_leaves = [x*300*(1-TEST_SIZE) for x in TRAINING_SIZES]
        # Calculate the labels, skipping half of them
        labels = [int(size / 3) if idx % 2 == 0 else "" for idx, size in enumerate(num_leaves)]

        axes[i].set_xticks(ticks=num_leaves,
                           labels=labels)
        axes[i].set_xlabel("Number of Leaves in Training Set")  # Update x-axis label

    axes[0].set_xlabel("")
    axes[2].set_xlabel("")
    axes[2].legend()
    plt.subplots_adjust(left=0.08, bottom=0.17, right=0.95)
    fig.savefig(f"{leaf}_validation_curves.pdf", format='pdf')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    make_all_leaf_learning_curves()
    # make_learning_curves_3_sensors("rice")
