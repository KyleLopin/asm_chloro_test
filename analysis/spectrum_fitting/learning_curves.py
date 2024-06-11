# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.model_selection import cross_validate, learning_curve, LearningCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, Normalizer,
                                   PolynomialFeatures, PowerTransformer,
                                   SplineTransformer, StandardScaler)
from sklearn.svm import SVR
plt.style.use("ggplot")

# local files
import get_data
from global_classes import CVScores
import preprocessors

CV = GroupShuffleSplit(test_size=0.2, n_splits=20)
# CV = ShuffleSplit(test_size=0.2, n_splits=20)
TRAINING_SIZES = [0.5, 0.7, 0.9, 1.0]
SCORING = "r2"
LINESTYLES = ['-', '--', ':', '-', '--', ':']
COLORS = ['black', 'blue', 'red', "green"]
# pipe1 = make_pipeline(StandardScaler(), PLSRegression(n_components=4))
# pipe2 = make_pipeline(RobustScaler(), PLSRegression(n_components=4))
# pipe1 = make_pipeline(RobustScaler(), PolynomialFeatures(degree=2), PLSRegression(n_components=5))
polynomial = FunctionTransformer(preprocessors.polynomial_expansion)

pipe1 = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=(1, 2),
                                         include_bias=False),
                      PLSRegression(n_components=10))

pipe2 = make_pipeline(StandardScaler(),
                      PolynomialFeatures(degree=(1, 2),
                                         include_bias=False),
                      LassoCV(max_iter=40000))
# pipe1 = make_pipeline(GroupAverager)

# pipe2 = make_pipeline(RobustScaler(), PolynomialFeatures(degree=2), LassoCV(max_iter=20000))
pipe3 = make_pipeline(StandardScaler(),
                      KernelPCA(n_components=6, kernel="poly", degree=2),
                      LinearRegression())


def make_learning_curves(ax: plt.Axes, x: pd.DataFrame,
                         y: pd.Series, groups: pd.Series, i: int,
                         regr: BaseEstimator=LinearRegression(),
                         score: str=SCORING):
    LearningCurveDisplay.from_estimator(regr, x, y, groups=groups,
                                        cv=CV, scoring=score,
                                        train_sizes=TRAINING_SIZES,
                                        ax=ax, std_display_style=None,
                                        line_kw={"ls": LINESTYLES[i],
                                                 "color": COLORS[i]})


def find_number_pls_latent_variables(max_comps: int = 6, poly_degrees: int = 1):
    _x, _y, _groups = get_data.get_x_y(sensor="as7262", leaf="mango",
                                       measurement_type="raw",
                                       int_time=50,
                                       send_leaf_numbers=True)
    _y = _y["Avg Total Chlorophyll (µg/cm2)"]
    cv = GroupShuffleSplit(test_size=0.2, n_splits=100)
    cv_scores = CVScores()
    if poly_degrees > 1:
        _x = PolynomialFeatures(degree=poly_degrees,
                                include_bias=False
                                ).fit_transform(_x)
    for i in range(1, max_comps+1):
        pls = PLSRegression(n_components=i, max_iter=5000)
        scores = cross_validate(pls, _x, _y, groups=_groups,
                                scoring='r2', cv=cv, return_train_score=True)
        cv_scores.add_scores(i, scores)
    cv_scores.plot()
    plt.show()


def make_learning_curves_graphs():
    fig, ax = plt.subplots()
    MEASUREMENT_TYPE = "reflectance"
    for i, regressor in enumerate([LinearRegression(),
                                   pipe1, pipe2, pipe3]):
        _x, _y, _groups = get_data.get_x_y(sensor="as7262", # led = "b'White'",
                                           leaf="mango",
                                           measurement_type=MEASUREMENT_TYPE,
                                           int_time=50,
                                           send_leaf_numbers=True)
        _y = _y["Avg Total Chlorophyll (µg/cm2)"]
        _x = preprocessors.snv(_x)
        # if i == 1:
        #     _x = preprocessors.polynomial_expansion(_x, standerize=True,
        #                                             degree=4)
        make_learning_curves(ax, x=_x, y=_y, groups=_groups,
                             i=i, regr=regressor)
    plt.ylim([0.6, 1.0])
    plt.show()


if __name__ == '__main__':
    make_learning_curves_graphs()
    # find_number_pls_latent_variables(poly_degrees=2, max_comps=20)
