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
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import cross_validate, GridSearchCV, ShuffleSplit

# local files
import get_data


# don't use
def _fit_model(sensor: str = "as7262", leaf: str = "banana",
              led: str = "White LED", current: str = "12.5 mA",
              integration_time: int = 150,
              measurement_type: str = "reflectance",
              take_mean=True):
    # data = get_data.get_data(sensor=sensor, leaf=leaf,
    #                          measurement_type=measurement_type,
    #                          mean=take_mean)
    # if leds:
    #     data = get_data.get_data_slices(df=data, selected_column="led",
    #                                     values=leds)
    # if integration_times:
    #     data = get_data.get_data_slices(df=data, selected_column="integration time",
    #                                     values=integration_times)
    # if currents:
    #     data = get_data.get_data_slices(df=data, selected_column="led current",
    #                                     values=currents)
    x, y = get_data.get_x_y(sensor=sensor, leaf=leaf, measurement_type=measurement_type,
                            led=led, led_current=current, int_time=integration_time,
                            mean=take_mean)

    print(x, y)

    model = LinearRegression()
    model.fit(x, y)
    y_predict = model.predict(x)
    plt.scatter(y, y_predict)
    plt.show()


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


def make_regr_table(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform regression analysis and generate tables of Mean Absolute Error (MAE)
    and R-squared (R2) scores. Test the 3 models of LinearRegression, Lasso,
    and partial least squares (with the best number of latent components found with gridsearch)
     on all the leaf types.

    Args:
        **kwargs (dict): dictionary of parameters to pass to the get_data.get_x_y function to
        get those conditions, i.e. integration time "int_time", and "led_current"

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the mean absolute errors
        and the r2 scores.  Each table have the columns of ["linear regression", "Lasso", "PLS"]
        for the method scores, and an index of ["mango", "banana", "jasmine", "rice", "sugarcane"]
        for the leaf species measured.

    """

    print(f"make regr table kwargs: {kwargs}")
    cross_validator = ShuffleSplit(n_splits=10, test_size=0.25)
    if kwargs["sensor"] == "as7265x":  # more channels needs more variables
        pls_parameters = {'n_components': np.arange(1, 12, 1)}
    else:
        pls_parameters = {'n_components': np.arange(1, 6, 1)}
    # find optimal number of latent variables for pls
    pls_grid_cv = GridSearchCV(PLS(), pls_parameters,
                               scoring='neg_mean_squared_error',
                               verbose=0, cv=cross_validator)
    # dictionary of regressors to test
    regressors = {"linear regression": LinearRegression(),
                  "Lasso": LassoCV(max_iter=10000),
                  "PLS": pls_grid_cv}
    # initialize empty DataFrames
    mae_table = pd.DataFrame(columns=regressors.keys())
    r2_table = pd.DataFrame(columns=regressors.keys())

    for name, regressor in regressors.items():
        for leaf in get_data.ALL_LEAVES:
            for _type in ["raw"]:
                x, y = get_data.get_x_y(leaf=leaf, measurement_type=_type,
                                        read_numbers=1, **kwargs)
                # print(x.shape)
                y = y["Avg Total Chlorophyll (µg/cm2)"]
                regressor.fit(x, y)
                # print(regressor.score(x, y))
                scores = cross_validate(regressor, x, y, cv=cross_validator,
                                        return_train_score=True,
                                        scoring=['neg_mean_absolute_error', 'r2'])
                mae_table.loc[leaf, name] = (
                    f"{-scores['train_neg_mean_absolute_error'].mean():.3f}\u00b1"
                    f"{-scores['train_neg_mean_absolute_error'].std():.3f}, "
                    f"{-scores['test_neg_mean_absolute_error'].mean():.3f}\u00b1"
                    f"{-scores['test_neg_mean_absolute_error'].std():.3f}")
                r2_table.loc[leaf, name] = (
                    f"{scores['train_r2'].mean():.3f}\u00b1"
                    f"{scores['train_r2'].std():.3f}, "
                    f"{scores['test_r2'].mean():.3f}\u00b1"
                    f"{scores['test_r2'].std():.3f}")
    return mae_table, r2_table


def make_excel_scoring_files():
    """ This function iterates through different sensors and creates Excel files containing MAE and R2 tables
    for various combinations of integration time and LED current.

    Excel files are saved with the following naming convention:
    - "<sensor>_mean_error.xlsx" for MAE tables.
    - "<sensor>_r2.xlsx" for R2 tables.

    Each Excel file contains multiple sheets corresponding to different combinations of integration time
    and LED current.

    Returns:
        None

    """
    for sensor in ["as7262", "as7263"]:
        with pd.ExcelWriter(f"{sensor}_mean_error.xlsx") as writer_mea:

            with pd.ExcelWriter(f"{sensor}_r2.xlsx") as writer_r2:
                for int_time in [50, 100, 150, 200, 250]:
                    for led_current in ["12.5 mA", "25 mA", "50 mA", "100 mA"]:
                        mae, r2 = make_regr_table(sensor=sensor,
                                                  int_time=int_time,
                                                  led_current=led_current)
                        mae.to_excel(writer_mea, sheet_name=f"sheet {int_time} msec {led_current}")
                        r2.to_excel(writer_r2, sheet_name=f"sheet {int_time} msec {led_current}")


if __name__ == '__main__':
    # fit_model(sensor="as7262")
    # regr = LinearRegression()
    # _cv = ShuffleSplit(n_splits=10, test_size=0.2)
    # get_mean_absolute(regr, _cv, sensor="as7262", int_time=150,
    #                   led_current="25 mA", leaf="banana",
    #                   measurement_type="raw")
    # make_regr_table(sensor="as7262")
    make_excel_scoring_files()
