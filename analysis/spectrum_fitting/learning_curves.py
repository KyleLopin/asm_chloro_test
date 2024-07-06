# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import (ARDRegression, HuberRegressor,
                                  LinearRegression, LassoLarsIC)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_validate, LearningCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer,
                                   PolynomialFeatures, StandardScaler)
plt.style.use("ggplot")

# local files
import get_data
from global_classes import CVScores
import preprocessors
from remove_outliers import remove_outliers_from_residues

CV = GroupShuffleSplit(test_size=0.2, n_splits=10)
CV = GroupShuffleSplit(test_size=0.2, n_splits=100)

TRAINING_SIZES = [0.5, 0.75, 0.9, 1.0]
SCORING = "r2"
LINESTYLES = ['-', '--', ':', '-', '-', ':', '-', '-', ':']
COLORS = ['black', 'blue', 'red', "green", 'magenta', 'cyan', 'darkred',
          'cyan', 'gold']
LEAF = "banana"
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
MEASUREMENT_TYPES = ["raw", "reflectance", "absorbance"]

polynomial = FunctionTransformer(preprocessors.polynomial_expansion)


pipe2 = make_pipeline(KernelPCA(n_components=20, kernel="rbf"),
                      LassoLarsIC(criterion='bic'))


grad_boost_depth=3
regressors = {
    "Linear Regression": LinearRegression(),

              "PLS": PLSRegression(n_components=10),
              # "Poly PLS (20 comps)": PLSRegression(n_components=20),

              "LassoCV BIC": LassoLarsIC(criterion='bic'),
              "Gradient Boost 50": GradientBoostingRegressor(n_estimators=50),
              "ARD": ARDRegression(lambda_2=0.001),
              "Hubur": HuberRegressor(max_iter=10000),
              # "Stacked ARD Gradient": StackingRegressor(
              #     [("ard", ARDRegression(lambda_2=0.001)),
              #      ("gradient", GradientBoostingRegressor(max_depth=2))]),
              # "Stacked Hubur Gradient": StackingRegressor(
              #     [("gradient", GradientBoostingRegressor(max_depth=2)),
              #      ("Hubur", HuberRegressor(max_iter=10000))]),
              # "Stacked ARD Hubur Gradient": StackingRegressor(
              #     [("ard", ARDRegression(lambda_2=0.001)),
              #      ("gradient", GradientBoostingRegressor(max_depth=2)),
              #      ("Hubur", HuberRegressor(max_iter=10000))]),
              # "Full stack": StackingRegressor(
              #     [("ard", ARDRegression(lambda_2=0.001)),
              #      ("gradient", GradientBoostingRegressor(n_estimators=50)),
              #      ("Hubur", HuberRegressor(max_iter=10000)),
              #      ("pls", PLSRegression(n_components=20, max_iter=10000))])
              }
# regressors = {"Linear Regression": LinearRegression()}


def make_learning_curves(ax: plt.Axes, x: pd.DataFrame,
                         y: pd.Series, groups: pd.Series, i: int,
                         name: str,
                         regr: BaseEstimator = LinearRegression(),
                         score: str = SCORING):
    LearningCurveDisplay.from_estimator(regr, x, y, groups=groups,
                                        cv=CV, scoring=score, n_jobs=-1,
                                        train_sizes=TRAINING_SIZES,
                                        ax=ax, std_display_style=None,
                                        line_kw={"color": COLORS[i],
                                                 "linestyle": LINESTYLES[i]})
    ax.plot([], [], label=name, color=COLORS[i], linestyle=LINESTYLES[i])


def find_number_pls_latent_variables(max_comps: int = 6, poly_degrees: int = 1):
    _x, _y, _groups = get_data.get_x_y(sensor="as7265x", leaf="mango",
                                       measurement_type="raw",
                                       int_time=50,
                                       send_leaf_numbers=True)
    _y = _y["Avg Total Chlorophyll (µg/cm2)"]
    cv = GroupShuffleSplit(test_size=0.2, n_splits=100)
    cv_scores = CVScores()
    if poly_degrees > 1:
        print("_x shape: ", _x.shape, poly_degrees)
        _x = PolynomialFeatures(degree=poly_degrees,
                                include_bias=False
                                ).fit_transform(_x)
        print("_x shape: ", _x.shape)
    for i in range(1, max_comps+1):
        pls = PLSRegression(n_components=i, max_iter=5000)
        scores = cross_validate(pls, _x, _y, groups=_groups,
                                scoring='r2', cv=cv, return_train_score=True)
        cv_scores.add_scores(i, scores)
    cv_scores.plot()
    plt.show()


def make_learning_curves_graph(leaf: str = LEAF, remove_outliers=False,
                               ax: plt.Axes|None=None, title: str="",
                               measurement_type: str="raw", sensor: str="as7262",
                               int_time: int=50, fit_poly: bool=True,
                               show_legend=True):
    if not ax:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title+" "+sensor)
    for i, (name, regressor) in enumerate(regressors.items()):
        print(name)
        if name == "PLS" and sensor in ['as7262', 'as7263']:
            regressor = PLSRegression(n_components=10)
        elif name == 'PLS':
            regressor = PLSRegression(n_components=20)
        _x, _y, _groups = get_data.get_x_y(sensor=sensor, led="White LED",
                                           leaf=leaf,
                                           measurement_type=measurement_type,
                                           int_time=int_time,
                                           send_leaf_numbers=True)
        _y = _y["Avg Total Chlorophyll (µg/cm2)"]
        if remove_outliers == "residues":
            _x, _y, _groups = remove_outliers_from_residues(_x, _y, _groups)
        if fit_poly:
            _x = PolynomialFeatures(degree=2).fit_transform(_x)
        _x = StandardScaler().fit_transform(_x)
        make_learning_curves(ax, x=_x, y=_y, groups=_groups,
                             i=i, name=name, regr=regressor)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # Filter out "Train" and "Test" entries
        filtered_handles = [handle for handle, label in zip(handles, labels) if
                            label not in ["Train", "Test"]]
        filtered_labels = [label for label in labels if label not in ["Train", "Test"]]
        plt.legend(filtered_handles, filtered_labels)


def make_all_leaf_learning_curves():
    for leaf in ALL_LEAVES:
        make_learning_curves_graph(leaf)

    plt.show()


def make_learning_curves_3_measure_modes(leaf:str, sensor: str="as7262",
                                         save_fig:bool=False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             sharey=True)
    show_legend=False
    fig.suptitle(leaf)
    for i, measurement_type in enumerate(MEASUREMENT_TYPES):
        if i == 2:
            show_legend = True

        make_learning_curves_graph(leaf, ax=axes[i],
                                   sensor=sensor,
                                   measurement_type=measurement_type,
                                   title=measurement_type,
                                   show_legend=show_legend)
    axes[0].set_xlabel("")
    axes[2].set_xlabel("")
    plt.tight_layout()


if __name__ == '__main__':
    # make_learning_curves_graph("banana")
    with PdfPages('all learning curves.pdf') as pdf:
        for sensor in ["as7262", "as7263", "as7265x"]:
            for leaf in ALL_LEAVES:

                make_learning_curves_3_measure_modes(leaf, sensor=sensor)
                pdf.savefig()

    # make_all_leaf_learning_curves()

    # find_number_pls_latent_variables(poly_degrees=2, max_comps=25)
