# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make an Anova tabe for AS7262, AS7263 and AS7265x sensor for different
led current, measurement types, and integration times for the different leave types
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import itertools

# installed libraries
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ARDRegression, HuberRegressor, LassoLarsIC, LinearRegression
from sklearn.model_selection import cross_val_score, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data


LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
MEASUREMENT_TYPES = ["raw", "reflectance", "absorbance"]
SENSORS = ["as7262", "as7263", "as7265x"]
PREPROCESS = ["Poly", "No Poly"]
SCORE = "r2"

regression_models_2_3 = {
    "ARD": ARDRegression(lambda_2=0.001),
    "Huber Regression": HuberRegressor(max_iter=10000),
    "Lasso IC": LassoLarsIC(criterion='bic'),
    "Linear Regression": LinearRegression(),
    "PLS": PLSRegression(n_components=10)
}

regression_models_5x = {
    "ARD": ARDRegression(lambda_2=0.01),
    "Huber Regression": HuberRegressor(max_iter=10000),
    "Lasso IC": LassoLarsIC(criterion='bic'),
    "Linear Regression": LinearRegression(),
    "PLS": PLSRegression(n_components=20)
}

cvs = {
    "Shuffle": GroupShuffleSplit(n_splits=5, test_size=0.2),
    "K Fold": GroupKFold(n_splits=5)
}


def make_anova_tables():
    # make csv file for each sensor and leave
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            # make Excel file
            filename = f"ANOVA/ANOVA {leaf} {sensor}.csv"
            make_anova_table(leaf, sensor)


def make_anova_table(leaf: str, sensor: str):

    combinations = itertools.product(MEASUREMENT_TYPES, INT_TIMES, LED_CURRENTS, PREPROCESS)

    for measure_type, int_time, current, preprocess in combinations:
        x, y, groups = get_data.get_x_y(
            leaf=leaf, sensor=sensor,
            measurement_type=measure_type,
            int_time=int_time, led_current=current,
            send_leaf_numbers=True)
        y = y["Avg Total Chlorophyll (µg/cm2)"]
        if preprocess == "Poly":
            x = PolynomialFeatures(degree=2).fit_transform(x)
        x = StandardScaler().fit_transform(x)
        if sensor == "as7262" or sensor == "as7263":
            regrs = regression_models_2_3
        elif sensor == "as7265x":
            regrs = regression_models_5x

        for regr_name, regr in regrs.items():
            for cv_name, cv in cvs.items():
                # get 5 test scores for each combo
                scores = cross_val_score(
                    regr, x, y, groups=groups, scoring=SCORE,
                    cv=cv)
                print(scores)


if __name__ == '__main__':
    make_anova_tables()
