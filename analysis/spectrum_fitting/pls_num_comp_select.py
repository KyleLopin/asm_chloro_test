# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Find the optimal number of latent variables for a partial least squared (PLS) regression
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data
import preprocessors
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

# plt.style.use('ggplot')
MIN_COMPONENTS = 1
USE_RFECV = False
USE_SFS = False
AIC_COLOR = "green"
AIC_FILL = "green"
# used to choose the number of components which is a fairly flat AIC score, and this
# will be repeated an additional N_SPLITS_OUTER for final AIC and R2 scores
N_SPLITS_INNER = 20  # To check graph layouts use lower numbers
N_SPLITS_OUTER = 50  # To reduce variance in final score use high numbers
BEST_AIC_COLOR = "black"

# This PLS scans output scan number and R2 and MAE scores, to not have to
# redo the tables all the time, use a CV seed to get reproducable data
# 124: R2=.962, MAE = 3.55
# 125: R2=.957 MAE=3.71
# 126; R2=.959 MAE=3.61  # use this one, middle of the 3 runs for MAE, R2 the same
RANDOM_STATE = 126  # for Mango runs is middle of for MAE


def calculate_aic(residual_sum_of_squares, num_observations, num_parameters):
    """
    Calculate the Akaike Information Criterion (AIC) for a regression model.

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
    aic : float
        The calculated AIC value, which incorporates model fit and complexity.
        A lower AIC value suggests a better model, balancing fit with simplicity.

    Notes
    -----
    The formula for AIC is:

        AIC = n * log(RSS / n) + 2 * k

    where:
        - n is the number of observations (num_observations)
        - RSS is the residual sum of squares
        - k is the number of parameters (num_parameters)
        - log is the natural logarithm

    A lower AIC indicates a model that better balances fit and simplicity.

    References
    ----------
    Akaike, Hirotugu. "A new look at the statistical model identification."
    *IEEE Transactions on Automatic Control* 19, no. 6 (1974): 716-723.
    doi:10.1109/TAC.1974.1100705
    """
    return num_observations * np.log(residual_sum_of_squares / num_observations) + \
        2 * num_parameters


def main_pls_selector_w_poly_analysis(leaf: str, sensor: str, ax, use_poly=False):
    """
    Perform PLS regression analysis for a given leaf and sensor,
    optionally using polynomial expansion.

    This function preprocesses the data, runs an outer cross-validation (CV) loop
    to evaluate PLS performance with varying numbers of latent variables, and plots
    the results on the given axes. Optionally, polynomial features can be included
    for expanded analysis.

    Parameters
    ----------
    leaf : str
        The name of the leaf to analyze.
    sensor : str
        The name of the sensor to use for data analysis (e.g., 'as7262', 'as7265x').
    ax : plt.Axes
        The primary Matplotlib Axes object for plotting R² scores.
    use_poly : bool, optional
        If True, performs polynomial feature expansion and evaluates the results.
        Default is False.

    Returns
    -------
    None
        The function modifies the provided Axes objects to display the results.

    Notes
    -----
    - The function uses different configurations for the AS7265x sensor due to its
      larger number of channels.
    - A secondary y-axis is created on the same plot to show AIC scores.
    - The analysis includes calculating mean MAE and R² scores for both original
      and polynomially expanded features.
    - Results are plotted using `plot_nested_cv_pls_variable_selector`.
    - Validation MAE and R² scores are plotted for each leaf/sensor combination

    """
    led = "White LED"
    max_comps = 6
    if sensor == "as7265x":
        led = "b'White IR'"
        max_comps = 14
    # x, y, groups = get_data.get_x_y(sensor=sensor, int_time=150,
    #                                 led_current="25 mA", leaf=leaf,
    #                                 measurement_type="absorbance",
    #                                 led=led,
    #                                 send_leaf_numbers=True)
    x, y, groups = get_data.get_cleaned_data(sensor, leaf)
    # y = y['Avg Total Chlorophyll (µg/cm2)']
    print(x.shape)
    y = y['Avg Total Chlorophyll (µg/cm2)']
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

    if use_poly:
        # x_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
        x_poly = preprocessors.polynomial_expansion(x, degree=2)
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
    ax.yaxis.get_label().set_fontsize(12)
    ax2.set_ylabel("AIC Score (lower is better)")
    # plt.title('Error as a function of the Number of PLS Components')
    ax.grid(axis='y')
    ax2.grid(False)


def plot_nested_cv_pls_variable_selector(results: dict, which_type: str,
                                         ax: plt.Axes, ax2: plt.Axes):
    """
    Plot the results of a nested cross-validation for PLS regression,
    showing R² scores and AIC scores with confidence intervals.

    Parameters
    ----------
    results : dict
        A dictionary containing the nested CV results with the following keys:
        - "r2_inner_means": Mean R² scores from the inner CV loop.
        - "r2_inner_std": Standard deviations of R² scores from the inner CV loop.
        - "aic_inner_means": Mean AIC scores from the inner CV loop.
        - "aic_inner_std": Standard deviations of AIC scores from the inner CV loop.
        - "outer_mae_scores": Mean absolute error scores from the outer CV loop.
        - "outer_r2_scores": R² scores from the outer CV loop.
    which_type : str
        Type of data being analyzed, either "original" or "poly" (polynomial features).
        Determines the plot color and line style.
    ax : plt.Axes
        The primary Matplotlib Axes object for plotting R² scores.
    ax2 : plt.Axes
        The secondary Matplotlib Axes object (twin y-axis) for plotting AIC scores.

    Returns
    -------
    None
        The function modifies the provided Axes objects to display the plots.

    Notes
    -----
    - The function visualizes R² scores as a line plot with a shaded region
      representing ±1 standard deviation for the inner CV loop.
    - AIC scores are plotted on a secondary y-axis (ax2) with similar visualization.
    - The optimal number of PLS components is determined based on the minimum AIC
      and annotated with a vertical dashed line.
    - Prints key metrics such as mean outer MAE and R² scores for reference.
    """
    # Plot the mean of the inner mean scores versus the
    # number of components with a shaded area for standard deviations
    max_comps = len(results["r2_inner_means"])
    # defaults if not polynomial fit
    # line_color = '#ff7f0e'
    line_color = "darkorange"
    fill_color = "darkorange"
    label = "Inner Test R² Scores"
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
        results["r2_inner_means"],
        # marker='o',
        color=line_color,
        label=label,
        ls=ls
    )

    # Shaded region for the standard deviation
    ax.fill_between(
        x_values,
        results["r2_inner_means"] - results["r2_inner_std"],
        results["r2_inner_means"] + results["r2_inner_std"],
        color=fill_color,
        alpha=0.3,
        label='±1 Standard Deviation'
    )
    ax2.plot(
        x_values,
        results["aic_inner_means"],
        # marker='o',
        color=AIC_COLOR,
        label="Akaike Information Criterion",
        # ls='dotted'
    )
    ax2.fill_between(
        x_values,
        results["aic_inner_means"] - results["aic_inner_std"],
        results["aic_inner_means"] + results["aic_inner_std"],
        color=AIC_FILL,
        alpha=0.3,
        label='±1 Standard Deviation'
    )

    # Determine the optimal component for the original data
    optimal_idx = np.argmin(results["aic_inner_means"])
    # optimal_idx = np.argmax(results["r2_inner_means"])
    # Since components are 1-based, but index is 0-based
    optimal_component = optimal_idx + MIN_COMPONENTS
    ax.axvline(optimal_component, color=BEST_AIC_COLOR, ls='dashed')
    print(f"optimal number of components = {optimal_component}")

    outer_mae = np.mean(results["outer_mae_scores"])
    outer_r2 = np.mean(results["outer_r2_scores"])
    print(f"mae = {outer_mae}, r2 = {outer_r2}")

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
    # ax.annotate(
    #     f'{start}\nValidation R²: {outer_r2:.2f}',
    #     xy_coords,
    #     xycoords="axes fraction",
    #     fontsize=10,
    #     color=line_color
    # )


def outer_cv_loop(x, y, groups, cv=None,
                  max_components: int = 6):
    """
    Perform outer cross-validation to evaluate the performance of a PLS regression model.
    Each outer fold performs an inner CV grid search to determine the optimal number of components,
    and the outer fold scores are collected to assess the model's generalization.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray
        Feature matrix containing the predictor variables.
    y : pd.Series or np.ndarray
        Target variable values.
    groups : pd.Series or np.ndarray
        Group labels for the samples used for stratified cross-validation.
    cv : cross-validation generator, optional
        Cross-validation splitting strategy. If None, a default `StratifiedGroupShuffleSplit`
        with specific settings will be used.
    max_components : int, optional
        The maximum number of PLS components to test. Defaults to 6.

    Returns
    -------
    dict
        A dictionary containing:
        - 'outer_mae_scores' : list of float
            Mean absolute error scores for each outer fold.
        - 'outer_r2_scores' : list of float
            R² scores for each outer fold.
        - 'inner_scores' : list of lists
            R² scores for each inner fold and number of components.
        - 'r2_inner_means' : np.ndarray
            Mean R² scores for each number of components across inner folds.
        - 'r2_inner_std' : np.ndarray
            Standard deviation of R² scores for each number of components across inner folds.
        - 'aic_inner_means' : np.ndarray
            Mean AIC scores for each number of components across inner folds.
        - 'aic_inner_std' : np.ndarray
            Standard deviation of AIC scores for each number of components across inner folds.
        - 'mae_inner_means' : np.ndarray
            Mean MAE scores for each number of components across inner folds.
        - 'mae_inner_std' : np.ndarray
            Standard deviation of MAE scores for each number of components across inner folds.

    Notes
    -----
    - The function uses a nested cross-validation approach where the outer loop evaluates
      model generalization and the inner loop determines the optimal number of components.
    - The best number of components for each outer fold is selected based on the minimum AIC score.
    - The outer loop results provide metrics to evaluate the model's performance, including
      mean absolute error (MAE) and R² scores.

    """
    # Use GroupShuffleSplit if no cross-validation strategy is provided
    if not cv:
        # cv = GroupShuffleSplit(n_splits=N_SPLITS_OUTER, test_size=0.20)
        cv = StratifiedGroupShuffleSplit(n_splits=N_SPLITS_OUTER, test_size=0.2,
                                         n_bins=10, random_state=RANDOM_STATE)
    # Initialize lists to store results
    outer_mae_scores = []
    outer_r2_scores = []
    inner_r2_per_component = []
    inner_r2_stds_per_component = []

    inner_mae_per_component = []
    inner_mae_stds_per_component = []

    aic_per_component = []
    aic_std_per_component = []

    # Iterate through the outer cross-validation splits
    for train_idx, test_idx in cv.split(x, y, groups=groups):
        # Split data into training and validation sets
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        # Perform inner CV Grid Search to find the optimal number of components
        # mean_inner_scores, std_inner_scores = grid_search_pls(x_train, y_train, groups_train,
        #                                                       max_components=max_components)
        inner_scores = grid_search_pls_aic_r2(x_train, y_train, groups_train,
                                              max_components=max_components)

        # Store the mean and std scores for the inner cross-validation for each component
        inner_r2_per_component.append(inner_scores["mean_r2"])
        inner_r2_stds_per_component.append(inner_scores["std_r2"])
        inner_mae_per_component.append(inner_scores["mean_mae"])
        inner_mae_stds_per_component.append(inner_scores["std_mae"])
        aic_per_component.append(inner_scores["mean_aic"])
        aic_std_per_component.append(inner_scores["std_aic"])
        # Use the best estimator from the inner CV for the outer validation
        best_component_idx = np.argmin(inner_scores["mean_aic"])
        # print("best n component ", best_component_idx+1)
        best_component = best_component_idx + MIN_COMPONENTS

        # Fit the final model using the best component from inner CV
        pls_best = PLSRegression(n_components=best_component, max_iter=10000)
        # Fit all of the training data, all 80 leaves / ~240 readings
        pls_best.fit(x_train, y_train)

        # Calculate the validation score for the outer fold
        y_pred = pls_best.predict(x_test)
        outer_mae_scores.append(mean_absolute_error(y_test, y_pred))
        outer_r2_scores.append(r2_score(y_test, y_pred))

    # Average the inner scores across all outer splits

    return {
        'outer_mae_scores': outer_mae_scores,
        'outer_r2_scores': outer_r2_scores,
        'inner_scores': inner_r2_per_component,
        'r2_inner_means': np.mean(inner_r2_per_component, axis=0),
        'r2_inner_std': np.mean(inner_r2_stds_per_component, axis=0),
        'aic_inner_means': np.mean(aic_per_component, axis=0),
        'aic_inner_std': np.mean(aic_std_per_component, axis=0),
        'mae_inner_means': np.mean(inner_mae_per_component, axis=0),
        'mae_inner_std': np.mean(inner_mae_stds_per_component, axis=0)
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
            - std_test_scores (list): Standard deviation of test scores
                                      for each number of components.
    """
    # Use GroupShuffleSplit if no cross-validation strategy is provided
    if not cv:
        # cv = GroupShuffleSplit(n_splits=N_SPLITS_INNER, test_size=0.25)
        cv = StratifiedGroupShuffleSplit(n_splits=N_SPLITS_INNER, test_size=0.25,
                                         n_bins=10, random_state=RANDOM_STATE)

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


def grid_search_pls_aic_r2(x, y, groups, cv=None, max_components: int = 6):
    """
    Perform a cross-validated grid search to find the optimal number of components
    for Partial Least Squares (PLS) regression using Akaike Information Criterion (AIC)
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
        dict: A dictionary containing lists of AIC and R^2 statistics
              for each number of components.
    """
    # Convert x to numpy array for consistent indexing
    x = np.array(x)
    y = np.array(y)

    if not cv:
        cv = StratifiedGroupShuffleSplit(n_splits=N_SPLITS_INNER, test_size=0.25,
                                         n_bins=10, random_state=RANDOM_STATE)

    # Initialize a dictionary to store results as lists
    results = {
        'n_components': [],
        'mean_aic': [],
        'std_aic': [],
        'mean_r2': [],
        'std_r2': [],
        'mean_mae': [],
        'std_mae': []
    }

    num_samples = x.shape[0]

    # Loop through each number of components to calculate AIC and R^2 scores
    for n_components in range(MIN_COMPONENTS, max_components + 1):
        pls = PLSRegression(n_components=n_components, max_iter=10000)
        aic_list = []
        r2_list = []
        mae_list = []

        if USE_RFECV:
            # Apply RFECV to select features within each CV split
            rfecv = RFECV(
                estimator=pls,
                step=1,
                cv=cv,  # Small CV within each fold to determine features
                scoring='neg_mean_absolute_error',
                min_features_to_select=max_components,
                n_jobs=-1
            )
            rfecv.fit(x, y, groups=groups)

            # print(f"optimal number of features {rfecv.n_features_}")

            # Use the selected features from RFECV for PLS
            x_fs = rfecv.transform(x)
        elif USE_SFS:
            sfs = SequentialFeatureSelector(
                estimator=pls,
                tol=0.005,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            x_fs = sfs.fit_transform(x, y, groups=groups)
        else:
            x_fs = x

        # Cross-validate using the provided CV strategy
        for train_idx, test_idx in cv.split(x_fs, y, groups=groups):
            # Use standard indexing for numpy arrays
            x_train, x_test = x_fs[train_idx], x_fs[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit the model
            pls.fit(x_train, y_train)
            y_pred = pls.predict(x_test)

            # Calculate Residual Sum of Squares (RSS)
            rss = np.sum((y_test - y_pred) ** 2)

            # Calculate AIC for this fold
            num_parameters = n_components + 1  # Components + intercept
            aic = calculate_aic(rss, num_samples, num_parameters)
            aic_list.append(aic)

            # Calculate R^2 score for this fold
            # r2 = r2_score(y_test, y_pred)
            # to calculate the r2 score after averaging the groups
            y_pred_grouped = pd.Series(y_pred, index=groups.iloc[test_idx]
                                       ).groupby('Leaf No.').mean()
            y_test_grouped = pd.Series(y_test, index=groups.iloc[test_idx]
                                       ).groupby('Leaf No.').mean()
            r2 = r2_score(y_test_grouped, y_pred_grouped)
            r2_list.append(r2)
            mae_list.append(mean_absolute_error(y_test, y_pred))

        # Compute the mean and standard deviation of AIC and R^2 across CV folds and
        # Store AIC and R^2 statistics for current number of components
        results['n_components'].append(n_components)
        results['mean_aic'].append(np.mean(aic_list))
        results['std_aic'].append(np.std(aic_list))
        results['mean_r2'].append(np.mean(r2_list))
        results['std_r2'].append(np.std(r2_list))
        results['mean_mae'].append(np.mean(mae_list))
        results['std_mae'].append(np.std(mae_list))

    return results


def plot_3_sensors(leaf: str):
    """
        Plot PLS hyperparameter tuning results for three sensors for a single leaf.

        This function generates a 3x1 grid of subplots, where each subplot shows the
        performance of PLS regression models for each sensor applied to the
        given leaf. It includes legends, axis labels, and a title to ensure clarity.

        Parameters
        ----------
        leaf : str
            The name of the leaf to analyze.

        Returns
        -------
        None
            Displays and saves the resulting figure as 'pls_scan.pdf'.

        Notes
        -----
        - Each subplot is generated using the `main_pls_selector_w_poly_analysis` function.
        - Y-axis limits are adjusted individually for the three sensors to improve clarity.
        - The legend, describing AIC scores and R² metrics, is placed in the bottom-right
          corner of the first subplot.
        - The function uses a fixed x-axis range (1 to 14) for consistency across sensors.
        """
    figure, axes = plt.subplots(3, 1, figsize=(7, 8.5),
                                sharex=True)
    for sensor, ax in zip(["as7262", "as7263", "as7265x"], axes):
        print(sensor)
        main_pls_selector_w_poly_analysis(leaf, sensor, ax,
                                          use_poly=False)
    # fill in legend with ax2 AIC label
    axes[0].plot([], [], label="Akaike Information\nCriterion Scores", color=AIC_COLOR)
    axes[0].fill_between([], [], label='±1 Standard Deviation', color=AIC_FILL, alpha=0.3)
    # to keep AIC as last entry put it here
    axes[0].plot([], [], label="Best AIC Score", color=BEST_AIC_COLOR, ls='--')

    axes[0].legend(fontsize=10, loc='lower right')
    axes[2].set_xlim([1, 14])
    axes[2].set_xlabel('Number of PLS Components')
    for i in [0, 2]:
        axes[i].set_ylim([0.8, 1.0])
    axes[1].set_ylim([0.7, 1.0])
    figure.suptitle(f"{leaf.capitalize()} PLS hyperparameter tuning",
                    fontsize=12)
    plt.tight_layout()
    figure.savefig("pls_scan.pdf", dpi=300, format='pdf')
    plt.show()


def plot_4_leaves_3_sensors(leaves):
    """
    Plot PLS hyperparameter tuning results for four leaves across three sensors.

    This function generates a 6x2 grid of subplots, where each subplot shows the
    performance of PLS regression models for a specific leaf-sensor combination.
    It adjusts the y-axis limits and adds annotations, legends, and titles for clarity.

    Parameters
    ----------
    leaves : list[str]
        A list of leaf names to include in the analysis. Each leaf is paired with
        results from three sensors: AS7262, AS7263, and AS7265x.

    Returns
    -------
    None
        Displays and saves the resulting figure as 'pls_scan_4_leaves.pdf'.

    Notes
    -----
    - Each subplot is generated using the `main_pls_selector_w_poly_analysis` function.
    - The function ensures appropriate y-axis limits based on the sensor and removes
      excessive y-axis labels for readability.
    - The legend describing the AIC score and R² metrics is placed in the bottom-right
      corner of the last row.
    """
    figure, axes = plt.subplots(6, 2, figsize=(7, 8.5),
                                sharex=True)
    axes = axes.flatten(order='F')
    print(axes)
    i = 0
    for leaf in leaves:
        for sensor in ["as7262", "as7263", "as7265x"]:
            main_pls_selector_w_poly_analysis(leaf, sensor, axes[i])
            if sensor in ["as7262", "as7265x"]:
                axes[i].set_ylim([0.7, 1.0])
            else:
                axes[i].set_ylim([0.6, 1.0])

            axes[i].annotate(f"{chr(i + 97)})", (0.1, 0.95), xycoords='axes fraction',
                             fontsize=12, fontweight='bold', va='top')

            if sensor == "as7262":
                # label leaf
                axes[i].annotate(leaf, (.6, .95), xycoords='axes fraction',
                                 fontsize=12, va='top')
            # well the subfunction added the ylabels for the 1 leaf figure, but not its too much
            # so turn them all off and then add text in the middle
            for axis in axes[i].figure.axes:
                axis.yaxis.label.set_visible(False)

            i += 1
            print(i, leaf, sensor)
    # and text to act as ylabels
    figure.text(0.04, 0.5, 'R²',
                ha='center', va='center', rotation='vertical', fontsize=12)
    figure.text(0.94, 0.5, "AIC Score (lower is better)",
                ha='center', va='center', rotation='vertical', fontsize=12)
    for j in [5, 11]:
        axes[j].set_xlabel('Number of PLS Components')
    figure.subplots_adjust(left=0.10, right=0.87, wspace=.31, bottom=0.07, top=0.94)
    figure.suptitle(f"PLS hyperparameter tuning",
                    fontsize=12, y=0.98)

    # fill in legend with ax2 AIC label
    axes[10].plot([], [], label="Akaike Information\nCriterion Scores", color=AIC_COLOR)
    axes[10].fill_between([], [], label='±1 Standard Deviation', color=AIC_FILL, alpha=0.3)
    # to keep AIC as last entry put it here
    axes[10].plot([], [], label="Best AIC Score", color=BEST_AIC_COLOR, ls='--')

    axes[10].legend(fontsize='xx-small', loc='lower right', frameon=False)

    figure.savefig("pls_scan_4_leaves.pdf", dpi=300, format='pdf')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # _x, _y, _groups = get_data.get_x_y(sensor="as7262", int_time=50,
    #                                    led_current="12.5 mA", leaf="mango",
    #                                    measurement_type="absorbance",
    #                                    send_leaf_numbers=True)
    # grid_search_pls(_x, _y, _groups)
    # main_pls_selector_w_poly_analysis("jasmine", "as7265x")
    # plot_3_sensors("mango")
    plot_4_leaves_3_sensors(["banana", "jasmine", "rice", "sugarcane"])
