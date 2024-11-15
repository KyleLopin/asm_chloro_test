# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
polynomial_expansion.py

This module provides a function for expanding the features of a
DataFrame by creating polynomial features.

Functions:
    polynomial_expansion: Expand the features of the input DataFrame
    by creating polynomial features up to the specified degree.

Usage:
    from polynomial_expansion import polynomial_expansion
    expanded_df = polynomial_expansion(input_df, degree=2)
"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import matplotlib.pyplot as plt  # for visualizing data
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import RegressorMixin
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data  # for visualizing data


def polynomial_expansion(x: pd.DataFrame,
                         degree: int = 2,
                         standerdize: bool = False
                         ) -> pd.DataFrame:
    """
    Expand the features of the input DataFrame `x` by creating polynomial features
    up to the specified `degree`.

    Args:
        x (pd.DataFrame): Input DataFrame containing the features.
        degree (int, optional): Degree of polynomial features to be created. Defaults to 2.
        standerdize (bool, optional): If a StandardScalar should be applied before the
        expanding the data.  Defaults to False

    Returns:
        pd.DataFrame: DataFrame with polynomial features added.

    Raises:
        ValueError: If `degree` is less than 2.

    Examples:
        >>> import pandas as pd
        >>> data = {'x1': [1, 2, 3], 'x2': [4, 5, 6]}
        >>> df = pd.DataFrame(data)
        >>> polynomial_expansion(df, degree=2)
           x1  x2  (x1)^2  (x2)^2
        0   1   4       1      16
        1   2   5       4      25
        2   3   6       9      36

        >>> polynomial_expansion(df, degree=3)
           x1  x2  (x1)^2  (x2)^2  (x1)^3  (x2)^3
        0   1   4       1      16       1      64
        1   2   5       4      25       8     125
        2   3   6       9      36      27     216

        >>> polynomial_expansion(df, degree=1)
        Traceback (most recent call last):
        ...
        ValueError: Input degrees needs to be 2 or larger, 1 is not valid
    """
    if degree < 2:
        raise ValueError(f"Input degrees needs to be 2 or larger, {degree} is not valid")
    print(x)
    if standerdize:
        x_scaled = StandardScaler().fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=x.columns)

    new_x = x.copy()

    for power in range(2, degree+1):
        for column in x.columns:
            new_x[f"({column})^{power}"] = x[column]**power
    return new_x


def snv(input_data) -> pd.DataFrame:
    """
    Standard Normal Variate function to preprocess a pandas DataFrame.

    Core code taken from:
    https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/

    Args:
        input_data (pandas.DataFrame or numpy.ndarray): Data to be corrected.

    Returns:
        pandas.DataFrame or numpy.ndarray: SNV corrected data. If a DataFrame was passed in,
        returns a DataFrame, otherwise, returns a numpy array.
    """
    _type = type(input_data)
    # Define a new array and populate it with the corrected data
    data_snv = np.zeros_like(input_data)
    _columns = None
    if type(input_data) is pd.DataFrame:
        _columns = input_data.columns
        _index = input_data.index
        input_data = input_data.to_numpy()

    for i in range(input_data.shape[0]):
        # Apply correction
        data_snv[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    if _type is pd.DataFrame:
        return pd.DataFrame(data_snv, columns=_columns, index=_index)
    # else return a numpy array
    return data_snv


def msc(input_data: pd.DataFrame, reference: pd.Series = None, return_reference: bool = False) -> (
        pd.DataFrame | np.ndarray | tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]):
    """
    Perform Multiplicative Scatter Correction (MSC).

    This function applies MSC to the input data, optionally using a provided
    reference spectrum. If no reference is given, the mean spectrum of the data
    is used as the reference.

    Reference: https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/

    Args:
        input_data (pandas.DataFrame or numpy.ndarray): Data to be corrected.
        reference (numpy.ndarray or pandas.Series, optional): Reference spectrum.
            If not provided, the reference will be estimated from the mean of
            the input data.
        return_reference (bool, optional): If True, returns the reference spectrum
            along with the MSC-corrected data. Defaults to False.

    Returns:
        pandas.DataFrame or numpy.ndarray, optional: The MSC-corrected data. If
        input_data is a DataFrame, returns a DataFrame; otherwise, returns a
        numpy array. If return_reference is True, also returns the reference
        spectrum as a second output.
    """
    # Determine the input data type
    _type = type(input_data)
    if _type is pd.DataFrame:
        _columns = input_data.columns
        _index = input_data.index
        input_data = input_data.to_numpy()

    # Mean center the correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()

    # Get the reference spectrum, estimating it from the mean if not provided
    if reference is None:
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Perform the MSC transformation
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Linear regression between the reference and current sample
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)
        # Apply the correction
        data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    # Convert back to a DataFrame if the input was a DataFrame
    if _type is pd.DataFrame:
        data_msc = pd.DataFrame(data_msc, columns=_columns, index=_index)
        ref = pd.Series(ref)

    # Return the corrected data, with or without the reference based on the flag
    if return_reference:
        return data_msc, ref
    return data_msc


def recursive_feature_elimination(x: pd.DataFrame, y: pd.Series,
                                  groups=None,
                                  regr: RegressorMixin = LinearRegression(),
                                  display_results: bool = True,
                                  cv=5):
    rfecv = RFECV(
        estimator=regr,
        step=1,
        cv=cv,
    )
    rfecv.fit(x, y, groups=groups)
    print(f"Optimal number of features: {rfecv.n_features_}")
    if display_results:
        print(rfecv)
        print("results")
        print(rfecv.cv_results_)
        n_steps = len(rfecv.cv_results_["mean_test_score"])
        plt.errorbar(
            range(1, n_steps + 1),
            rfecv.cv_results_["mean_test_score"],
            yerr=rfecv.cv_results_["std_test_score"],
        )
        plt.ylim([0.1, 1.0])
        plt.show()


def ard_regression_grid_search(x, y, groups, title):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        x, y, groups)
    logspace_values = np.logspace(-8, 1, num=10)

    param_grid = {
        'alpha_1': np.logspace(-10, 5, num=10),
        'alpha_2': np.logspace(-10, 6, num=10),
        'lambda_1': np.logspace(-10, 5, num=10),
        'lambda_2': np.logspace(-10, 4, num=10),
    }
    # Initialize the ARDRegression model
    ard_reg = ARDRegression()
    cv = GroupShuffleSplit(test_size=0.2, n_splits=10)
    # Perform grid search
    grid_search = GridSearchCV(ard_reg, param_grid,
                               cv=cv, scoring='r2',
                               n_jobs=-1)
    grid_search.fit(x, y, groups=groups)

    # Extract results
    results = grid_search.cv_results_

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Initialize a dictionary to store the best scores for each hyperparameter
    best_scores = {'alpha_1': [], 'alpha_2': [], 'lambda_1': [], 'lambda_2': []}

    # Extract best scores for each hyperparameter value
    for param in best_scores.keys():
        for value in param_grid[param]:
            mask = results_df[f'param_{param}'] == value
            best_score = results_df[mask]['mean_test_score'].max()
            best_scores[param].append((value, best_score))

    # Convert best scores to a DataFrame for plotting
    best_scores_df = {param: pd.DataFrame(scores, columns=[param, 'best_score']) for param, scores
                      in best_scores.items()}
    plt.figure(figsize=(10, 6))

    # Plot each hyperparameter's best score
    for param, df in best_scores_df.items():
        plt.plot(df[param], df['best_score'], label=param, marker='o')
        plt.xscale('log')  # Set the x-axis to logarithmic scale

    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Best R2 Score')
    plt.title('Best R2 Score for Each Hyperparameter Value (Log Scale)')
    plt.legend()
    plt.grid(True)
    # Best parameters and model
    best_params = grid_search.best_params_
    print("Best parameters found:", best_params)
    best_params_text = "\n".join([f"{key}: {value:.0e}" for key, value in best_params.items()])
    plt.annotate(f'Best Parameters:\n{best_params_text}', xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    best_ard_reg = grid_search.best_estimator_

    # Predict on the test set with the best model
    y_pred_best = best_ard_reg.predict(X_test)

    # Calculate and print the Mean Squared Error for the best model
    mse_best = r2_score(y_test, y_pred_best)
    print(f'Mean Squared Error of the best model: {mse_best}')

    # Initialize and fit the ARDRegression model with default hyperparameters
    default_ard_reg = ARDRegression()
    default_ard_reg.fit(X_train, y_train)

    # Predict on the test set with the default model
    y_pred_default = default_ard_reg.predict(X_test)

    # Calculate and print the Mean Squared Error for the default model
    mse_default = r2_score(y_test, y_pred_default)
    print(f'Mean Squared Error of the default model: {mse_default}')
    # plt.ylim([0.91, 0.94])
    plt.show()


if __name__ == '__main__':
    # _x = pd.DataFrame([[1, 2], [3, 4]], columns=["450 nm", "500 nm"])
    #
    # _x = polynomial_expansion(_x)
    # print('===')
    # print(_x)
    sensor = "as7262"
    leaf = "mango"
    measure_type = "raw"
    int_time = 100
    x, y, groups = get_data.get_x_y(sensor=sensor,  leaf=leaf,
                                    measurement_type=measure_type,
                                    int_time=int_time,
                                    send_leaf_numbers=True)
    x = PolynomialFeatures(degree=2).fit_transform(x)
    y = y["Avg Total Chlorophyll (µg/cm2)"]
    ard_regression_grid_search(x, y, groups)

    # # x, _ = msc(x)
    # _index = x.index
    # poly = PolynomialFeatures(degree=2)
    # poly.fit(x)
    # _columns = poly.get_feature_names_out()
    # x = poly.fit_transform(x)
    # x = StandardScaler().fit_transform(x)
    # x = pd.DataFrame(x, columns=_columns, index=_index)
    # # x = polynomial_expansion(x, degree=4, standerdize=True)
    # print(f"x shape: {x.shape}")
    # # plt.plot(x.T)
    # # plt.show()
    # cv=GroupShuffleSplit(test_size=0.2, n_splits=100)
    # recursive_feature_elimination(x, y, groups=groups,
    #                               cv=cv)
