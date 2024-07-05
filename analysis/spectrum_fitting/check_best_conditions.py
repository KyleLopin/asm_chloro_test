# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make an Anova tabe for AS7262, AS7263 and AS7265x sensor for different
led current, measurement types, and integration times for the different leave types
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import itertools
import warnings

# installed libraries
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ARDRegression, HuberRegressor, LassoLarsIC, LinearRegression
from sklearn.model_selection import cross_val_score, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
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


def make_anova_table_files():
    # make csv file for each sensor and leave
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            make_anova_table_file(leaf, sensor)


def make_anova_table_file(leaf: str, sensor: str):
    print(f"making {leaf} {sensor} ANOVA table")
    results = []
    int_times = INT_TIMES
    if sensor == "as7265x":
        int_times = [50, 100, 150]
    combinations = itertools.product(MEASUREMENT_TYPES, int_times, LED_CURRENTS, PREPROCESS)

    for measure_type, int_time, current, preprocess in combinations:
        x, y, groups = get_data.get_x_y(
            leaf=leaf, sensor=sensor,
            measurement_type=measure_type,
            int_time=int_time, led_current=current,
            send_leaf_numbers=True)
        y = y["Avg Total Chlorophyll (µg/cm2)"]
        regrs = regression_models_2_3
        if sensor == "as7265x":  # change if as7265x sensor
            regrs = regression_models_5x
        if preprocess == "Poly":
            x = PolynomialFeatures(degree=2).fit_transform(x)
        else:
            if sensor in ["as7262", "as7263"]:
                regrs["PLS"] = PLSRegression(n_components=6)
            elif sensor == "as7265x":
                regrs["PLS"] = PLSRegression(n_components=18)
        x = StandardScaler().fit_transform(x)

        for regr_name, regr in regrs.items():
            for cv_name, cv in cvs.items():
                # get 5 test scores for each combo
                scores = cross_val_score(
                    regr, x, y, groups=groups, scoring=SCORE,
                    cv=cv)
                # Append the results to the list
                for score in scores:
                    results.append({
                        "Leaf": leaf,
                        "Sensor": sensor,
                        "Measurement Type": measure_type,
                        "Integration Time": int_time,
                        "LED Current": current,
                        "Preprocess": preprocess,
                        "Regression Model": regr_name,
                        "Cross Validation": cv_name,
                        "Score": score
                    })

                # Create a DataFrame from the results list
            results_df = pd.DataFrame(results)

            # Save the DataFrame to a CSV file
            filename = f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv"
            results_df.to_csv(filename, index=False)


def print_pg_anova_tables():
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            print_pg_anova_table(leaf, sensor)


def print_pg_anova_table(leaf: str, sensor: str):
    filename =f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv"
    df = pd.read_csv(filename)

    # Perform the ANOVA test
    aov = pg.anova(dv='Score',
                   between=['Measurement Type', 'Integration Time', 'LED Current',
                            'Preprocess', 'Regression Model', 'Cross Validation'],
                   data=df)


def check_leaf_sensor_p_values():

    combined_data = []
    # make combined dataset
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            df = pd.read_csv(f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv")
            df['Leaf'] = leaf
            df['Sensor'] = sensor
            combined_data.append(df)
    print('======')
    combined_df = pd.concat(combined_data, ignore_index=True)
    # Convert relevant columns to float32
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[numeric_cols] = combined_df[numeric_cols].astype(np.float32)
    print(combined_df)
    # Perform two-way ANOVA
    # aov = pg.anova(dv='Score', between=['Leaf', 'Sensor'], data=combined_df,
    #                detailed=True)
    # doesnt work too much memory
    aov = pg.anova(dv='Score',
                   between=['Leaf', 'Sensor', 'Regression Model',
                            'Cross Validation', 'Preprocess',
                            "LED Current", "Integration Time",
                            "Measurement Type"],
                   data=combined_df,
                   detailed=True)
    # Display the ANOVA table
    print(aov)


if __name__ == '__main__':
    # make_anova_table_files()
    check_leaf_sensor_p_values()
