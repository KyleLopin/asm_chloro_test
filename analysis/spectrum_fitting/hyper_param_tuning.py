# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import itertools

# installed libraries
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import PolynomialFeatures

# local files
import get_data

LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
MEASUREMENT_TYPES = ["raw", "reflectance", "absorbance"]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
CV = GroupShuffleSplit(test_size=0.2, n_splits=10)

def get_best_parameters(X, y, groups, param_grid, scoring='neg_mean_squared_error', test_size=0.2,
                        cv_splits=5, random_state=42):
    """
    Get the best parameters for ARDRegression using GroupShuffleSplit cross-validation
    and calculate the mean squared error for the best and default parameters.

    Parameters:
    - X: Features.
    - y: Target.
    - groups: Group labels for the data.
    - param_grid: Dictionary of hyperparameters to search.
    - scoring: Scoring method for GridSearchCV (default is 'neg_mean_squared_error').
    - test_size: Proportion of the data to include in the test split (default is 0.2).
    - cv_splits: Number of cross-validation splits (default is 5).
    - random_state: Random state for reproducibility (default is 42).

    Returns:
    - best_params: Dictionary of best parameters.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=test_size,
        random_state=random_state)

    # Define the GroupShuffleSplit cross-validator
    group_shuffle_split = GroupShuffleSplit(n_splits=cv_splits, test_size=test_size,
                                            random_state=random_state)

    # Initialize the ARDRegression model
    ard_reg = ARDRegression()

    # Perform grid search with GroupShuffleSplit
    grid_search = GridSearchCV(ard_reg, param_grid, cv=group_shuffle_split, scoring=scoring,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train, groups=groups_train)

    # Extract the best parameters and estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    # Predict on the test set with the best model
    y_pred_best = best_estimator.predict(X_test)
    mse_best = mean_squared_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    # Initialize and fit the ARDRegression model with default hyperparameters
    default_ard_reg = ARDRegression()
    default_ard_reg.fit(X_train, y_train)

    # Predict on the test set with the default model
    y_pred_default = default_ard_reg.predict(X_test)
    mse_default = mean_squared_error(y_test, y_pred_default)
    r2_default = r2_score(y_test, y_pred_default)

    # Print the results
    print("Best parameters found:", best_params)
    print(f'Mean Squared Error of the best model: {mse_best}')
    print(f'Mean Squared Error of the default model: {mse_default}')
    print(f'R2 of the best model: {r2_best}')
    print(f'R2 of the default model: {r2_default}')

    return best_params


def save_all_grid_searches(type:str = "ARD"):
    matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
    for sensor in ["as7262", "as7263"]:
        pdf_filename = f"ARD grid wide search {sensor}.pdf"
        with PdfPages(pdf_filename) as pdf:
            combinations = itertools.product(ALL_LEAVES, MEASUREMENT_TYPES, INT_TIMES, LED_CURRENTS)

            for leaf, measure_type, int_time, current in combinations:
                x, y, groups = get_data.get_x_y(
                    leaf=leaf, sensor=sensor,
                    measurement_type=measure_type,
                    int_time=int_time, led_current=current,
                    send_leaf_numbers=True)
                y = y["Avg Total Chlorophyll (µg/cm2)"]
                title = f"leaf: {leaf}, {measure_type}, int time: {int_time}, current: {current}"
                print(title)
                if type == "ARD":
                    make_ard_grid_searches(x, y, groups, title, pdf)


def make_ard_grid_searches(x: pd.DataFrame, y: pd.Series,
                           groups: pd.Series, title: str,
                           pdf: PdfPages):
    param_grid = {
        'alpha_1': np.logspace(-10, 5, num=10),
        'alpha_2': np.logspace(-10, 6, num=10),
        'lambda_1': np.logspace(-10, 5, num=10),
        'lambda_2': np.logspace(-10, 4, num=10),
    }
    ard_reg = ARDRegression()
    # Perform grid search
    grid_search = GridSearchCV(ard_reg, param_grid,
                               cv=5, scoring='r2',
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
    plt.title(f'Best R2 Score for Each Hyperparameter Value (Log Scale)\n{title}')
    plt.legend()
    plt.grid(True)
    # Best parameters and model
    best_params = grid_search.best_params_
    print("Best parameters found:", best_params)
    best_params_text = "\n".join([f"{key}: {value:.0e}" for key, value in best_params.items()])
    plt.annotate(f'Best Parameters:\n{best_params_text}', xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    pdf.savefig()
    plt.close()


if __name__ == '__main__':
    save_all_ard_grid_searches()
    # x, _y, groups = get_data.get_x_y(sensor="as7262", leaf="mango",
    #                                measurement_type="raw",
    #                                int_time=50,
    #                                send_leaf_numbers=True)
    # y = _y["Avg Total Chlorophyll (µg/cm2)"]
    # x = PolynomialFeatures(degree=2).fit_transform(x)
    # # Define a range of values from 1e-8 to 1 in logarithmic scale
    # logspace_values = np.logspace(-8, 2, num=15)
    #
    # param_grid = {
    #     'alpha_1': logspace_values,
    #     'alpha_2': logspace_values,
    #     'lambda_1': logspace_values,
    #     'lambda_2': logspace_values,
    # }
    #
    # # Assuming X_train, y_train, and groups_train are already defined
    # best_params = get_best_parameters(x, y, groups, param_grid,
    #                                   scoring='r2')
    #
    # print("Best parameters found:", best_params)
