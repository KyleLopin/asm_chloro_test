# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ARDRegression, LassoLarsIC

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler

# local files
import get_data
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit

pls_n_comps = {"as7262": {"banana": 6, "jasmine": 4, "mango": 5, "rice": 6, "sugarcane": 6},
               "as7263": {"banana": 5, "jasmine": 5, "mango": 5, "rice": 5, "sugarcane": 5},
               "as7265x": {"banana": 8, "jasmine": 14, "mango": 11, "rice": 6, "sugarcane": 6}}

LEAVES = ["banana", "jasmine", "mango", "rice", "sugarcane"]

def plot_model_analysis(X, y, models, cv=5, scoring='neg_mean_squared_error', groups=None,
                        column_names=None, **kwargs):
    """
    Plots the coefficients and learning curves for a list of regression models.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - models: List of tuples (name, model) where name is a string for the model name
              and model is a scikit-learn estimator.
    - cv: Cross-validation splitting strategy (default is 5-fold).
    - scoring: Scoring method to use for the learning curve (default is 'neg_mean_squared_error').
    - groups: Group labels for the samples used while splitting the dataset into training/test set.
    - column_names: List of column names for labeling the x-axis (default is None, using feature indices).
    - **kwargs: Additional keyword arguments for learning_curve.
    """
    # Initialize plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cv = StratifiedGroupShuffleSplit(test_size=.2, n_splits=20, n_bins=10)

    # Bar width for side-by-side placement
    bar_width = 0.25
    num_models = len(models)

    # Loop over models
    for i, (name, model) in enumerate(models):
        # Fit the model
        model.fit(X, y)

        # Get coefficients, if available
        try:
            coef = model.coef_
            print(model)
            print(model.coef_.shape)
            print(type(model.coef_))
            if coef.shape[0] == 1:
                coef = coef.flatten()
                print('flattening')
                print(coef.shape)
            # Shift bar positions to make them side by side
            positions = np.arange(len(coef)) + i * bar_width
            ax1.bar(positions, coef, bar_width, alpha=0.7, label=f'{name}')
        except AttributeError as e:
            print(f"Model {name} does not have coefficients to plot.")
            print(e)

        # Compute learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring, groups=groups, **kwargs
        )

        # Calculate mean and standard deviation
        train_scores_mean = -train_scores.mean(
            axis=1) if scoring == 'neg_mean_squared_error' else train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(
            axis=1) if scoring == 'neg_mean_squared_error' else test_scores.mean(axis=1)

        # Plot learning curves
        ax2.plot(train_sizes, train_scores_mean, 'o-', label=f'{name} - Train')
        ax2.plot(train_sizes, test_scores_mean, 'o-', label=f'{name} - Test', linestyle='--')

    # Format the coefficients plot
    ax1.set_title('Model Coefficients')
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Coefficient Value')

    # Set x-ticks with feature names or indices
    if column_names is not None:
        ax1.set_xticks(np.arange(len(column_names)) + (num_models - 1) * bar_width / 2)
        ax1.set_xticklabels(column_names, rotation=45, ha='right')
    else:
        ax1.set_xticks(np.arange(len(coef)) + (num_models - 1) * bar_width / 2)
        ax1.set_xticklabels(np.arange(len(coef)))

    ax1.legend()

    # Format the learning curves plot
    ax2.set_title('Learning Curves')
    ax2.set_xlabel('Training Examples')
    ax2.set_ylabel('Score' if scoring != 'neg_mean_squared_error' else 'Mean Squared Error')
    ax2.legend()

    # Layout adjustment
    plt.tight_layout()
    plt.show()


def view_model_analysis():
    sensor = "as7265x"
    leaf = "rice"
    measurement_type = "absorbance"
    int_time = 50
    led = "White LED"
    if sensor == "as7265x":
        led = "b'White IR'"

    _x, _y, _groups = get_data.get_x_y(sensor=sensor, led=led,
                                       leaf=leaf,
                                       measurement_type=measurement_type,
                                       int_time=int_time,
                                       led_current="12.5 mA",
                                       send_leaf_numbers=True)
    print(_x)
    feature_names = _x.columns
    print('===')
    # _y = _y["Avg Total Chlorophyll (µg/cm2)"]
    _y = _y["Avg Total Chlorophyll (µg/cm2)"]
    print(_x.shape)
    if False:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        _x = poly.fit_transform(_x)
        feature_names = poly.get_feature_names_out(feature_names)
        print(feature_names)
    print(_x.shape)
    _x = StandardScaler().fit_transform(_x)

    # Models to analyze
    models = [
        ('pls', PLSRegression(n_components=10)),
        ("ARD", ARDRegression(lambda_2=0.001)),
        ("LassoCV BIC", LassoLarsIC(criterion='bic'))]

    plot_model_analysis(_x, _y, models=models, scoring="r2",
                        train_sizes=[.7, .9, 1.0], groups=_groups,
                        column_names=feature_names)


def plot_leaf_model_coeffs(sensor: str, axis: Axes) -> None:
    """
    Fit a model for each leaf and plot the model coefficients as a bar chart on the provided axis.

    Parameters
    ----------
    sensor : str
        The name of the sensor to use for data retrieval.
    axis : matplotlib.axes.Axes
        The axis on which to plot the model coefficients.

    Returns
    -------
    None
        This function modifies the provided axis and does not return a value.
    """
    bar_width = 5  # Width of each bar
    num_leaves = len(LEAVES)
    offsets = [i * bar_width for i in range(num_leaves)]  # Calculate offsets for each leaf

    for i, leaf in enumerate(LEAVES):
        # Get cleaned data
        x, y, _ = get_data.get_cleaned_data(sensor, leaf, mean=False)
        y = y["Avg Total Chlorophyll (µg/cm2)"]
        wavelengths = x.columns
        x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]

        # Standardize the features
        x = StandardScaler().fit_transform(x)

        # Fit the PLS model
        model = PLSRegression(n_components=pls_n_comps[sensor][leaf])
        model.fit(x, y)

        # Flatten model coefficients to 1D
        coefficients = model.coef_.flatten()

        # Offset the bar positions for this leaf
        x_positions = [w + offsets[i] - (bar_width / 2) for w in x_wavelengths]
        # Create a bar chart
        axis.bar(x_positions, coefficients, width=bar_width, label=f"Leaf {leaf}")

    # Customize the axis
    # axis.set_title(f"Model Coefficients for {sensor}")
    # axis.set_xlabel("Wavelength (nm)")
    # axis.set_ylabel("Coefficient Value")
    # axis.legend()


def plot_all_sensors_coeffs(sensors: list[str]) -> None:
    """
    Create a figure with 1 column and 3 rows to plot model coefficients
    for each sensor, using the `plot_leaf_model_coeffs` function.

    Parameters
    ----------
    sensors : list[str]
        A list of sensor names to process.

    Returns
    -------
    None
        The function creates and displays a figure with plots.
    """
    # Create a figure with 1 column and 3 rows
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)

    for sensor, ax in zip(sensors, axes):
        # Call the previously defined function to plot on the current axis
        plot_leaf_model_coeffs(sensor, ax)

    axes[0].legend()
    # Adjust layout to prevent overlapping of subplots
    fig.tight_layout()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # plot_all_sensors_coeffs(["as7262", "as7263", "as7265x"])
    x_columns = None
    for sensor in ["as7262", "as7263", "as7265x"]:
        x_full = pd.DataFrame()  # Initialize an empty DataFrame for features
        y_full = pd.Series(dtype=float)  # Initialize an empty Series for targets

        for leaf in LEAVES:
            # Get cleaned data
            x, y, _ = get_data.get_cleaned_data(sensor, leaf, mean=False)
            y = y["Avg Total Chlorophyll (µg/cm2)"]  # Extract the target column
            x_columns = x.columns  # Save the original column names

            # Standardize the features while keeping them as a DataFrame
            x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x_columns, index=x.index)

            # Concatenate the current leaf's data to the full dataset
            x_full = pd.concat([x_full, x])
            y_full = pd.concat([y_full, y])

        # Output for debugging or further processing
        print(f"Sensor: {sensor}")
        print("Combined X shape:", x_full.shape)
        print("Combined Y shape:", y_full.shape)
        model = PLSRegression(n_components=6)
        if sensor == "as7265x":
            model = PLSRegression(n_components=10)
        model.fit(x, y)
        plt.bar(x_columns, model.coef_.flatten())
        plt.show()
