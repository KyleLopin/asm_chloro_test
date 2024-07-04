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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ARDRegression, HuberRegressor, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


def save_all_grid_searches(regr_type:str = "ARD", show_figures=True):
    if not show_figures:
        matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
    for sensor in ["as7262"]:
        pdf_filename = f"{regr_type} grid wide search {sensor}.pdf"
        with PdfPages(pdf_filename) as pdf:
            combinations = itertools.product(ALL_LEAVES, MEASUREMENT_TYPES, INT_TIMES, LED_CURRENTS)

            for leaf, measure_type, int_time, current in combinations:
                x, y, groups = get_data.get_x_y(
                    leaf=leaf, sensor=sensor,
                    measurement_type=measure_type,
                    int_time=int_time, led_current=current,
                    send_leaf_numbers=True)
                y = y["Avg Total Chlorophyll (µg/cm2)"]
                x = PolynomialFeatures(degree=2).fit_transform(x)
                x = StandardScaler().fit_transform(x)
                title = f"leaf: {leaf}, {measure_type}, int time: {int_time}, current: {current}"
                print(title)
                if regr_type == "ARD":
                    make_ard_grid_searches(x, y, groups, title, pdf)
                elif regr_type == "Huber":
                    param_grid = {
                        'epsilon': [0.5, 1, 1.35, 1.55, 1.7, 2.0, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10],
                        'alpha': np.logspace(-5, 1, num=10),
                    }

                    # Initialize the Huber Regressor
                    huber = HuberRegressor(max_iter=1000)
                    make_regr_grid_search_best_params(
                        huber, param_grid, x, y, groups, title, pdf,
                        show_figure=show_figures)
                elif regr_type == "GradientBoost":
                    make_grad_boost_search(x, y, groups, title, pdf)
                elif regr_type == "PLS":
                    param_grid = {"n_components": [3, 5, 7, 8, 9, 10, 13, 16, 20]}
                    pls = PLSRegression(max_iter=10000)
                    make_regr_grid_search_best_params(
                        pls, param_grid, x, y, groups, title, pdf,
                        show_figure=show_figures
                    )
                elif regr_type == "Lasso":
                    param_grid = {"alpha": np.logspace(-4, -1, 10)}
                    lasso = Lasso(max_iter=10000)
                    make_regr_grid_search_best_params(
                        lasso, param_grid, x, y, groups, title, pdf,
                        show_figure=show_figures
                    )
                elif regr_type == "Kernel":
                    param_grid = {"pca__n_components": [5, 10, 13, 16, 20],
                                  'pca__gamma': np.logspace(-6, -1, 10)
                                  }
                    kernel = Pipeline([("pca", KernelPCA(kernel='rbf')),
                                       ('LR', LinearRegression())])
                    make_regr_grid_search_best_params(
                        kernel, param_grid, x, y, groups, title, pdf,
                        show_figure=show_figures
                    )


def make_grad_boost_search(x, y, groups, title, pdf):
    # Define the parameter grid
    param_grid = {
        # 'loss': ['huber', 'squared_error'],
        # 'learning_rate': [0.01, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5],
        # 'n_estimators': [10, 20, 50, 100, 150, 200],
        # 'subsample': [0.1, 0.3, 0.45, 0.6, 0.8, 1.0],
        # 'criterion': ['friedman_mse', 'squared_error'],
        # 'min_samples_split': [2, 5, 10, 20, 30],
        # 'min_samples_leaf': [1, 2, 4, 6, 10],
        # 'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 1, 2, 3, 4, 5, 7, 10],
        # 'validation_fraction': [0.1, 0.2, 0.3],
        # "ccp_alpha": [0.1, 0.2, 0.3, 0.5, 1]
        # 'min_impurity_decrease': [0.0, 0.01, 0.1],
        # 'max_features': ['sqrt', 'log2', None],
        # 'max_features': [1, 2, 3, 5, 10, 15, 20, 30],
        # 'max_leaf_nodes': [None, 10, 20, 30, 100, 1000]
    }

    # Initialize the Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid,
                               cv=CV, verbose=1, n_jobs=-1)

    # Fit the grid search to your data
    grid_search.fit(x, y, groups=groups)

    # Extract the results
    results = grid_search.cv_results_
    print(results)
    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Visualize the best result for each hyperparameter
    def plot_best_results(param_name, param_values):
        means = []
        stds = []
        for value in param_values:
            mask = np.array([str(param) == str(value) for param in results[f'param_{param_name}']])
            mean_test_scores = np.nan_to_num(results['mean_test_score'][mask], nan=0.0)
            std_test_scores = np.nan_to_num(results['std_test_score'][mask], nan=0.0)

            mean_test_score = mean_test_scores.max()
            std_test_score = std_test_scores.max()

            means.append(mean_test_score)
            stds.append(std_test_score)
        param_values = [x if x is not None else 0 for x in param_values]
        print(param_values)
        print(means)
        plt.figure(figsize=(10, 5))
        plt.errorbar(param_values, means, yerr=stds, fmt='o', capsize=5)
        plt.title(f'Performance for different {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Mean Test Score')
        plt.grid(True)
        plt.show()

    # Plot best results for each hyperparameter
    for param_name in param_grid.keys():
        param_values = param_grid[param_name]
        plot_best_results(param_name, param_values)

    # Print the best parameters and best score
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

def make_regr_grid_search_best_params(
        regr, param_grid: dict, x: pd.DataFrame, y: pd.Series,
        groups: pd.Series, title: str,
        pdf: PdfPages, show_figure: bool = True):

    # Perform grid search
    grid_search = GridSearchCV(
        regr, param_grid, cv=CV, scoring='r2', n_jobs=-1)
    grid_search.fit(x, y, groups=groups)
    # Extract results
    results = grid_search.cv_results_

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    # Initialize a dictionary to store the best scores for each hyperparameter
    # Initialize a dictionary to store the best scores for each hyperparameter
    best_scores = {key: [] for key in param_grid}
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
    if show_figure:
        plt.show()
    else:
        pdf.savefig()
        plt.close()


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
                               cv=CV, scoring='r2',
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
    # save_all_grid_searches("ARD")
    # save_all_grid_searches("Huber", show_figures=False)
    save_all_grid_searches("PLS", show_figures=True)
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
