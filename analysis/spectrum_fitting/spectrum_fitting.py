# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Initial functions developed to fit the spectrum from the as7262, as7263 and as7265x sensor
that measured leaf spectrum to calculate their chlorophyll.

Not all functions are working
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin  # for type-hinting
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import (cross_val_predict, cross_validate,
                                     GridSearchCV, ShuffleSplit, GroupShuffleSplit)
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

# local files
import get_data
LED_CURRENT = "12.5 mA"
INT_TIME = 50
CV = GroupShuffleSplit(n_splits=20, test_size=0.2)


def _get_mean_absolute(regr, cv, **kwargs):  # don't use
    print(kwargs)
    x, y = get_data.get_x_y(**kwargs)

    scores = cross_validate(regr, x, y, cv=cv, return_train_score=True,
                            scoring='neg_mean_absolute_error')
    print(scores)
    print(f"test score: {scores['test_score'].mean()} +- {scores['test_score'].std()}")
    print(f"train score: {scores['train_score'].mean()} +- {scores['train_score'].std()}")
    return (scores['test_score'].mean(), scores['test_score'].std(),
            scores['train_score'].mean(), scores['train_score'].std())



def graph_y_predicted_vs_y_actual(
        x=None, y=None, groups=None,
        regressor: RegressorMixin = LinearRegression, cv: int=5,
        y_column: str = "Avg Total Chlorophyll (µg/cm2)",
        **kwargs):
    """ Graph the predicted chlorophyll versus actual chlorophyll levels

    Args:
        regressor (sklearn regression model): which regressor to fit the data to
        cv (sklearn cross-validator): which cross-validator to us
        y_column (str): which chlorophyll column to use, total, ch a or ch b
        sensor (str): which sensor to use
        **kwargs: arguements to pass to get_data.get_x_y to get the proper
        data subset.

    Returns:

    """
    print(x)
    if x is None:
        x, y, groups = get_data.get_x_y(**kwargs)
        y = y[y_column]
    y_predict = cross_val_predict(regressor, x, y, cv=5,
                                  groups=groups)
    scores = cross_validate(regressor, x, y, cv=cv,
                            groups=groups, return_train_score=True,
                            scoring='neg_mean_absolute_error')
    scores = [f"{scores['test_score'].mean()} \u00B1 {scores['test_score'].std()}",
              f"{scores['train_score'].mean()} \u00B1 {scores['train_score'].std()}"]
    print(scores)
    plt.scatter(y, y_predict)
    # x = StandardScaler().fit_transform(x)
    # plt.plot(x.T)  # sanity check
    plt.show()


if __name__ == '__main__':
    # fit_model(sensor="as7262")
    # regr = LinearRegression()
    _cv = 5
    _cv = GroupShuffleSplit(n_splits=20, test_size=0.2)
    lasso = LassoCV(max_iter=10000)
    pls = PLSRegression(n_components=12)
    svr = SVR()
    estimators = [
        ('pls', pls),

        ('lasso', lasso)
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression()
    )


    # get_mean_absolute(regr, _cv, sensor="as7262", int_time=150,
    #                   led_current="25 mA", leaf="banana",
    #                   measurement_type="raw")
    # make_regr_table(sensor="as7262")
    # make_anova_excel_files()
    # graph_y_predicted_vs_y_actual(regressor=regr, cv=_cv,
    #                               sensor="as7262", int_time=150,
    #                               led_current="25 mA", leaf="banana",
    #                               measurement_type="raw", mean=True)
    _x, _y, _groups = get_data.get_x_y(
        sensor="as7262", leaf="mango", measurement_type="reflectance",
        int_time=100, led_current="12.5 mA", send_leaf_numbers=True)
    _y = _y['Avg Total Chlorophyll (µg/cm2)']
    _x = PolynomialFeatures(degree=2).fit_transform(_x)
    print("make graph")
    # graph_y_predicted_vs_y_actual(
    #     x=_x, y=_y, groups=_groups,
    #     regressor=stacking_regressor, cv=_cv)
    pls.fit(_x, _y)
    print(pls.score(_x, _y))
    _y_pre = pls.predict(_x)
    threshold = 20

    mask_high = _y_pre > threshold
    mask_low = _y_pre < threshold
    x_high = _x[mask_high]
    y_high = _y[mask_high]

    pls.fit(x_high, y_high)
    print(pls.score(x_high, y_high))
    x_low = _x[mask_low]
    y_low = _y[mask_low]

    pls.fit(x_low, y_low)
    print(pls.score(x_low, y_low))
