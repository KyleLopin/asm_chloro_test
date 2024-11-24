# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Find the optimal number of latent variables for a partial least squared (PLS) regression
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data
import preprocessors
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

plt.style.use('ggplot')
MIN_COMPONENTS = 2


def neg_exp(x):
    return np.exp(-x)
    # return np.exp(x)


def neg_log(x):
    return -np.log(x)
    # return np.log(x)


def calculate_bic(residual_sum_of_squares, num_observations, num_parameters):
    """
    Calculate the Bayesian Information Criterion (BIC) for a regression model.

    Parameters
    ----------
    residual_sum_of_squares : float
        The residual sum of squares (RSS) from the model, which measures the
        difference between observed and predicted values.

    num_observations : int
        The total number of observations (data points) used in the model.

    num_parameters : int
        The number of estimated parameters in the model, including the intercept.

    Returns
    -------
    bic : float
        The calculated BIC value, which incorporates model fit and complexity.
        A lower BIC value suggests a better model, balancing fit with simplicity.

    Notes
    -----
    The formula for BIC is:

        BIC = n * log(RSS / n) + k * log(n)

    where:
        - n is the number of observations (num_observations)
        - RSS is the residual sum of squares
        - k is the number of parameters (num_parameters)
        - log is the natural logarithm

    A lower BIC indicates a model that better balances fit and simplicity.

    References
    ----------
    Schwarz, Gideon E. "Estimating the dimension of a model." *The Annals of Statistics*
    6, no. 2 (1978): 461-464. doi:10.1214/aos/1176344136
    """
    return num_observations * np.log(residual_sum_of_squares / num_observations) + \
           num_parameters * np.log(num_observations)


def main_pls_selector_w_poly_analysis(leaf: str, sensor: str, ax):
    """
    Perform the main analysis by comparing original data with polynomial-transformed data.


    Args:
        leaf (str): The name of the leaf sample being analyzed.
        sensor (str): The sensor type used for data collection.
        ax: pyplot axis

    Returns:

    """
    led = "White LED"
    max_comps = 6
    if sensor == "as7265x":
        led = "b'White IR'"
        max_comps = 18
    x, y, groups = get_data.get_x_y(sensor=sensor, int_time=50,
                                    led_current="12.5 mA", leaf=leaf,
                                    measurement_type="absorbance",
                                    led=led,
                                    send_leaf_numbers=True)
    # y = y['Avg Total Chlorophyll (µg/cm2)']
    y = y['Avg Chlorophyll a (µg/cm2)']
    x_ss = StandardScaler().fit_transform(x)
    results = outer_cv_loop(pd.DataFrame(x_ss), y, groups,
                            max_components=max_comps)
    ax2 = ax.twinx()
    plot_nested_cv_pls_variable_selector(results, "original",
                                         ax, ax2)
    mae_score = np.mean(results["outer_mae_scores"])
    r2_score = np.mean(results["outer_r2_scores"])
    print(r2_score)
    print(mae_score)

    x_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    x_poly = StandardScaler().fit_transform(x_poly)

    print("poly results")
    max_comps = 12
    if sensor == "as7265x":
        max_comps = 24
    results = outer_cv_loop(pd.DataFrame(x_poly), y, groups,
                            max_components=max_comps)
    mae_score = np.mean(results["outer_mae_scores"])
    r2_score = np.mean(results["outer_r2_scores"])
    print(r2_score)
    print(mae_score)
    plot_nested_cv_pls_variable_selector(results, "poly",
                                         ax, ax2)
    # Labeling and grid
    ax.set_ylabel('R²')
    ax2.set_label("BIC (lower is better)")
    # plt.title('Error as a function of the Number of PLS Components')
    ax.grid(True)
    ax2.grid(False)


def plot_nested_cv_pls_variable_selector(results: dict, which_type: str,
                                         ax: plt.Axes, ax2: plt.Axes):
    # Plot the mean of the inner mean scores versus the number of components with a shaded area for std
    max_comps = len(results["mean_inner_means"])
    # defaults if not polynomial fit
    line_color = '#ff7f0e'
    fill_color = "orange"
    label = "Sensor data"
    xy_coords = (0.3, 0.28)
    ls = 'solid'
    start = "Original"
    if which_type == "poly":
        line_color = "green"
        fill_color = "green"
        label = "Polynomial Sensor data"
        xy_coords = (0.3, 0.1)
        start = "Polynomial"
        ls = '--'

    x_values = range(MIN_COMPONENTS, max_comps+MIN_COMPONENTS)
    # Line plot for the mean of inner means
    ax.plot(
        x_values,
        results["mean_inner_means"],
        # marker='o',
        color=line_color,
        label=label,
        ls=ls
    )

    # Shaded region for the standard deviation
    ax.fill_between(
        x_values,
        results["mean_inner_means"] - results["mean_inner_stds"],
        results["mean_inner_means"] + results["mean_inner_stds"],
        color=fill_color,
        alpha=0.3,
        label='±1 Standard Deviation'
    )
    ax2.plot(
        x_values,
        results["bic_inner_means"],
        marker='o',
        color=line_color,
        label=label,
        ls=ls
    )
    ax2.fill_between(
        x_values,
        results["bic_inner_means"] - results["bic_inner_std"],
        results["bic_inner_means"] + results["bic_inner_std"],
        color=fill_color,
        alpha=0.3,
        label='±1 Standard Deviation'
    )

    # Determine the optimal component for the original data
    optimal_idx = np.argmax(results["mean_inner_means"])
    # Since components are 1-based, but index is 0-based
    optimal_component = optimal_idx + MIN_COMPONENTS
    print(f"optimal number of components = {optimal_component}")
    outer_mae = np.mean(results["outer_mae_scores"])
    outer_r2 = np.mean(results["outer_r2_scores"])


    # Mark and annotate the optimal point
    # ax.scatter(
    #     optimal_component,
    #     results["mean_inner_means"][optimal_idx],
    #     marker='x',
    #     color=fill_color,
    #     edgecolor='black',
    #     s=100,
    #     zorder=5,
    #     label='Optimal Components'
    # )
    ax.annotate(
        f'{start}\nValidation R²: {outer_r2:.2f}',
        xy_coords,
        xycoords="axes fraction",
        fontsize=10,
        color=line_color
    )


def outer_cv_loop(x, y, groups, cv=None,
                    max_components: int = 6):
    """
    Perform outer cross-validation to evaluate the performance of a PLS regression model.
    Each fold performs an inner CV grid search to find the optimal number of components,
    and the outer fold scores are collected to assess the model's generalization.

    Args:
        x (pd.DataFrame or np.ndarray): Feature matrix containing the predictor variables.
        y (pd.Series or np.ndarray): Target variable values.
        groups (pd.Series or np.ndarray): Group labels for the samples used for cross-validation.
        max_components (int, optional): The maximum number of PLS components to test. Defaults to 6.
        outer_splits (int, optional): The number of outer cross-validation splits. Defaults to 10.

    Returns:
        dict: A dictionary containing:
            - outer_scores (list): Validation scores for each outer fold.
            - inner_scores (list): Mean test scores for each inner fold.
            - std_scores (list): Standard deviation of test scores for each inner fold.
    """
    # Use GroupShuffleSplit if no cross-validation strategy is provided
    if not cv:
        cv = GroupShuffleSplit(n_splits=20, test_size=0.20)
        cv = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.2, n_bins=10)
    # Initialize lists to store results
    outer_mae_scores = []
    outer_r2_scores = []
    inner_r2_per_component = []
    inner_r2_stds_per_component = []
    bic_per_component = []
    bic_std_per_component = []

    # Iterate through the outer cross-validation splits
    for train_idx, test_idx in cv.split(x, y, groups=groups):
        # Split data into training and validation sets
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        # Perform inner CV Grid Search to find the optimal number of components
        # mean_inner_scores, std_inner_scores = grid_search_pls(x_train, y_train, groups_train,
        #                                                       max_components=max_components)
        inner_scores = grid_search_pls_bic_r2(x_train, y_train, groups_train,
                                              max_components=max_components)

        # Store the mean and std scores for the inner cross-validation for each component
        inner_r2_per_component.append(inner_scores["mean_r2"])
        inner_r2_stds_per_component.append(inner_scores["std_r2"])
        bic_per_component.append(inner_scores["mean_bic"])
        bic_std_per_component.append(inner_scores["std_bic"])
        # Use the best estimator from the inner CV for the outer validation
        best_component_idx = np.argmin(inner_scores["mean_bic"])
        print("best n component ", best_component_idx+1)
        best_component = best_component_idx + MIN_COMPONENTS

        # Fit the final model using the best component from inner CV
        pls_best = PLSRegression(n_components=best_component, max_iter=10000)
        pls_best.fit(x_train, y_train)

        # Calculate the validation score for the outer fold
        y_pred = pls_best.predict(x_test)
        outer_mae_scores.append(mean_absolute_error(y_test, y_pred))
        outer_r2_scores.append(r2_score(y_test, y_pred))

    # Average the inner scores across all outer splits
    r2_inner_means = np.mean(inner_r2_per_component, axis=0)
    r2_inner_stds = np.mean(inner_r2_stds_per_component, axis=0)
    bic_inner_means = np.mean(bic_per_component, axis=0)
    bic_inner_std = np.mean(bic_std_per_component, axis=0)

    return {
        'outer_mae_scores': outer_mae_scores,
        'outer_r2_scores': outer_r2_scores,
        'inner_scores': inner_r2_per_component,
        'mean_inner_means': r2_inner_means,
        'mean_inner_stds': r2_inner_stds,
        'bic_inner_means': bic_inner_means,
        'bic_inner_std': bic_inner_std
    }


# Use as the inner nested CV parameter search
def grid_search_pls(x, y, groups, cv=None,
                    max_components: int = 6):
    """
    Perform a cross-validated grid search to find the optimal number of components
    for Partial Least Squares (PLS) regression.

    Args:
        x (pd.DataFrame or np.ndarray): Feature matrix containing the predictor variables.
        y (pd.Series or np.ndarray): Target variable values.
        groups (pd.Series or np.ndarray): Group labels for the samples used for cross-validation.
                                          This allows splitting data based on group information.
        cv (cross-validation generator, optional): Cross-validation splitting strategy.
                                                   Defaults to `GroupShuffleSplit` with 20 splits
                                                   and a test size of 0.25.
        max_components (int, optional): The maximum number of PLS components to test.
                                        Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - mean_test_scores (list): Mean test scores for each number of components.
            - std_test_scores (list): Standard deviation of test scores for each number of components.
    """
    # Use GroupShuffleSplit if no cross-validation strategy is provided
    if not cv:
        cv = GroupShuffleSplit(n_splits=20, test_size=0.25)
        cv = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.25, n_bins=10)

    # Initialize PLS regression model with a high iteration limit to ensure convergence
    pls = PLSRegression(max_iter=10000)
    # pls = TransformedTargetRegressor(
    #     regressor=pls, func=neg_log, inverse_func=neg_exp)

    # Define the parameter grid for PLS (number of components)
    param_grid = {
        'n_components': range(MIN_COMPONENTS, max_components+1)
    }

    # Initialize GridSearchCV with PLS model, parameter grid, and cross-validation strategy
    clf = GridSearchCV(
        pls,
        param_grid,
        cv=cv,
        scoring='r2',
        refit=True
    )

    clf.fit(x, y, groups=groups)
    # Return mean and standard deviation of test scores for each component

    return clf.cv_results_["mean_test_score"], clf.cv_results_["std_test_score"]


def grid_search_pls_bic_r2(x, y, groups, cv=None, max_components: int = 6):
    """
    Perform a cross-validated grid search to find the optimal number of components
    for Partial Least Squares (PLS) regression using Bayesian Information Criterion (BIC)
    and R^2 score.

    Args:
        x (pd.DataFrame or np.ndarray): Feature matrix containing the predictor variables.
        y (pd.Series or np.ndarray): Target variable values.
        groups (pd.Series or np.ndarray): Group labels for the samples used for cross-validation.
                                          This allows splitting data based on group information.
        cv (cross-validation generator, optional): Cross-validation splitting strategy.
                                                   Defaults to `GroupShuffleSplit` with 20 splits
                                                   and a test size of 0.25.
        max_components (int, optional): The maximum number of PLS components to test.
                                        Defaults to 6.

    Returns:
        dict: A dictionary containing lists of BIC and R^2 statistics
              for each number of components.
    """
    # Convert x to numpy array for consistent indexing
    x = np.array(x)
    y = np.array(y)

    if not cv:
        cv = StratifiedGroupShuffleSplit(n_splits=20, test_size=0.25, n_bins=10)

    # Initialize a dictionary to store results as lists
    results = {
        'n_components': [],
        'mean_bic': [],
        'std_bic': [],
        'mean_r2': [],
        'std_r2': []
    }

    num_samples = x.shape[0]

    # Loop through each number of components to calculate BIC and R^2 scores
    for n_components in range(MIN_COMPONENTS, max_components + 1):
        pls = PLSRegression(n_components=n_components, max_iter=10000)
        bic_list = []
        r2_list = []

        # Cross-validate using the provided CV strategy
        for train_idx, test_idx in cv.split(x, y, groups=groups):
            # Use standard indexing for numpy arrays
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit the model
            pls.fit(x_train, y_train)
            y_pred = pls.predict(x_test)

            # Calculate Residual Sum of Squares (RSS)
            rss = np.sum((y_test - y_pred) ** 2)

            # Calculate BIC for this fold
            num_parameters = n_components + 1  # Components + intercept
            bic = calculate_bic(rss, num_samples, num_parameters)
            bic_list.append(bic)

            # Calculate R^2 score for this fold
            r2 = r2_score(y_test, y_pred)
            r2_list.append(r2)

        # Compute the mean and standard deviation of BIC and R^2 across CV folds
        mean_bic = np.mean(bic_list)
        std_bic = np.std(bic_list)
        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)

        # Store BIC and R^2 statistics for current number of components
        results['n_components'].append(n_components)
        results['mean_bic'].append(mean_bic)
        results['std_bic'].append(std_bic)
        results['mean_r2'].append(mean_r2)
        results['std_r2'].append(std_r2)

    return results


def plot_3_sensors(leaf: str):
    figure, axes = plt.subplots(3, 1, figsize=(7, 8.5),
                                 sharex=True)
    for sensor, ax in zip(["as7262", "as7263", "as7265x"], axes):
        print(sensor)
        main_pls_selector_w_poly_analysis(leaf, sensor, ax)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[2].set_xlabel('Number of PLS Components')
    for i in [0, 2]:
        axes[i].set_ylim([0.85, 1.0])
    figure.suptitle(f"{leaf.capitalize()} PLS hyperparameter tuning", fontsize=20)
    # figure.savefig("pls_scan.pdf", dpi=300, format='pdf')
    plt.show()


if __name__ == '__main__':
    # _x, _y, _groups = get_data.get_x_y(sensor="as7262", int_time=50,
    #                                    led_current="12.5 mA", leaf="mango",
    #                                    measurement_type="absorbance",
    #                                    send_leaf_numbers=True)
    # grid_search_pls(_x, _y, _groups)
    # main_pls_selector_w_poly_analysis("jasmine", "as7265x")
    plot_3_sensors("mango")
