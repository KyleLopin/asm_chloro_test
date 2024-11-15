# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Helper Functions for Data Filtering and Model Prediction

This module provides utility functions for various tasks, including:
1. Filtering a DataFrame based on matching values from a selected row in another DataFrame.
2. Performing a single train-test split for machine learning model predictions.

Functions:
- filter_df: Filters rows in a DataFrame based on a selected row from another DataFrame, excluding specified columns.
- single_split_predict: Performs a single train-test split and returns predictions for both the training and testing sets.

Usage Example:
The module includes an example usage scenario if executed directly, demonstrating:
1. Filtering rows in a DataFrame based on a selected row from another DataFrame.
2. Displaying the filtered results.
"""

__author__ = "Kyle Vitautas Lopin"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# local files
from global_classes import GroupedData


def filter_df(selected_row, df, exclude_columns=["Score"]):
    """
    Filters the DataFrame df based on the values in the selected_row,
    excluding specified columns.

    Args:
    - selected_row (pd.Series): A row from the DataFrame df1.
    - df (pd.DataFrame): The DataFrame to be filtered.
    - exclude_columns (list of str): Columns to exclude from the matching process.

    Returns:
    - pd.DataFrame: The filtered DataFrame containing rows that match the selected_row values
      in the specified columns.
    """
    # Get the columns to match on, excluding the specified columns
    columns_to_match = [col for col in df.columns if col not in exclude_columns]

    # Create a filter condition for each column to match on
    filter_condition = pd.Series([True] * len(df))

    for col in columns_to_match:
        filter_condition &= (df[col] == selected_row[col])

    return df[filter_condition]


def single_split_predict(model, x, y, test_size=0.2, random_state=None):
    """
    Perform a single train-test split, fit the model on the training data,
    and return the predictions for both testing and training sets.

    Parameters:
    - model: The machine learning model to use for fitting.
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: Target vector (numpy array or pandas Series).
    - test_size: Fraction of the dataset to use for testing (default is 0.2).
    - random_state: Controls the shuffling applied to the data before splitting.

    Returns:
    - test_pred: Predictions for the testing set.
    - train_pred: Predictions for the training set.
    """
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    # Clone the model to ensure it's a fresh instance
    model = clone(model)

    # Fit the model on the training data
    model.fit(x_train, y_train)

    # Predict on both the training and testing data
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    print(train_pred)
    print(test_pred)

    return test_pred, train_pred


def group_based_train_test_split(data: GroupedData, test_size=0.2, random_state=None, n_splits=1):
    """
    Splits the input GroupedData into training and testing sets based on groups,
    yielding either a single split or multiple splits based on n_splits.

    Args:
    - data (GroupedData): The data to be split, containing features, target, and group information.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int, optional): Controls the randomness of the split for reproducibility.
    - n_splits (int): The number of splits to generate (default is 1).

    Returns:
    - tuple[GroupedData, GroupedData]: A single training and testing split if n_splits is 1.
    - generator: A generator yielding multiple (GroupedData, GroupedData) splits if n_splits > 1.
    """
    def _group_based_train_test_split():
        """
        An inner helper function that yields training and testing GroupedData for the specified number of splits.
        """
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)
        for train_idx, test_idx in splitter.split(data.x, data.y, groups=data.group):
            # Create GroupedData instances for training and testing sets
            train_data = GroupedData(data.x.iloc[train_idx], data.y.iloc[train_idx], data.group.iloc[train_idx])
            test_data = GroupedData(data.x.iloc[test_idx], data.y.iloc[test_idx], data.group.iloc[test_idx])
            yield train_data, test_data

    # Call the inner function to either return one split or the generator for multiple splits
    if n_splits == 1:
        # Return the first split
        return next(_group_based_train_test_split())
    else:
        # Return the generator for multiple splits
        return _group_based_train_test_split()


def predict_train_test_with_grouping(
    x: pd.DataFrame,
    y: pd.Series,
    group: pd.Series,
    regr: RegressorMixin,
    test_size: float = 0.2,
    random_state: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train a regression model using the training set and make predictions on both
    the training and test sets. Group the predicted and true values by the specified group
    and return the results as DataFrames.

    Args:
    - x (pd.DataFrame): Features for the regression model.
    - y (pd.Series): Target values.
    - group (pd.Series): Group labels used for splitting the data.
    - regr (RegressorMixin): A regression model that implements fit and predict methods.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int, optional): Controls the randomness of the split for reproducibility.

    Returns:
    - tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the grouped train and test sets.
      Each DataFrame has columns for the actual values, predicted values, and group labels,
      with the results grouped by the group identifier.
    """
    # Split the data into training and testing sets while preserving group information
    train_data, test_data = group_based_train_test_split(
        GroupedData(x, y, group),
        test_size=test_size,
        random_state=random_state
    )

    # Fit the regression model on the training data
    regr.fit(train_data.x, train_data.y)

    # Generate predictions for both the training and testing sets
    train_preds = regr.predict(train_data.x)
    test_preds = regr.predict(test_data.x)

    # Create DataFrames for the training set with actual, predicted values, and groups
    train_df = pd.DataFrame({
        'y_true': train_data.y,
        'y_pred': train_preds,
        'Group': train_data.group
    }).groupby('Group').mean()

    # Create DataFrames for the testing set with actual, predicted values, and groups
    test_df = pd.DataFrame({
        'y_true': test_data.y,
        'y_pred': test_preds,
        'Group': test_data.group
    }).groupby('Group').mean()

    return train_df, test_df


def evaluate_model_scores(x: pd.DataFrame, y: pd.Series,
                          groups: pd.Series, n_splits=10,
                          regressor=None, test_size=0.1,
                          group_by_mean=False):
    """
    Evaluate the regression model using GroupShuffleSplit and return statistical scores for both training and testing sets.

    Args:
        x (pd.DataFrame): Feature matrix.
        y (pd.Series): Target values.
        groups (pd.Series): Group labels corresponding to each data point.
        n_splits (int): Number of times to perform the split and evaluate the model.
        regressor: The regression model to use (if None, defaults to LinearRegression).
        test_size (float): Proportion of the dataset to include in the test split.
        group_by_mean (bool): Whether to group the predictions by the group labels and calculate the mean before scoring.

        Returns:
        dict: A dictionary with the following structure:
            {
                "train": {
                    "r2": (average R^2 score, standard deviation R^2),
                    "mae": (average Mean Absolute Error, standard deviation MAE)
                },
                "test": {
                    "r2": (average R^2 score, standard deviation R^2),
                    "mae": (average Mean Absolute Error, standard deviation MAE)
                }
            }
    """
    # Initialize lists to store scores
    train_r2_scores = []
    train_mae_scores = []
    test_r2_scores = []
    test_mae_scores = []

    # Create the GroupShuffleSplit object outside the loop
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=None)

    for train_idx, test_idx in splitter.split(x, y, groups):
        # Train and test data
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the regressor on the training data
        if regressor is None:
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()

        regressor.fit(x_train, y_train)

        # Make predictions on the training and test data
        y_train_pred = regressor.predict(x_train)
        y_test_pred = regressor.predict(x_test)

        # Optionally group by the group labels and calculate the mean for each group
        if group_by_mean:
            # Convert predictions to pandas Series with the original group labels as the index
            y_train_pred = pd.Series(y_train_pred.flatten(), index=y_train.index
                                     ).groupby('group').mean()
            y_test_pred = pd.Series(y_test_pred.flatten(), index=y_test.index
                                    ).groupby('group').mean()
            # Update the true values accordingly
            y_train = y_train.groupby('group').mean()
            y_test = y_test.groupby('group').mean()

        # Calculate R^2 score and Mean Absolute Error for training data
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)

        # Calculate R^2 score and Mean Absolute Error for test data
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Append scores to the respective lists
        train_r2_scores.append(train_r2)
        train_mae_scores.append(train_mae)
        test_r2_scores.append(test_r2)
        test_mae_scores.append(test_mae)

    # Calculate averages and standard deviations for training scores
    avg_train_r2 = np.mean(train_r2_scores)
    std_train_r2 = np.std(train_r2_scores)
    avg_train_mae = np.mean(train_mae_scores)
    std_train_mae = np.std(train_mae_scores)

    # Calculate averages and standard deviations for test scores
    avg_test_r2 = np.mean(test_r2_scores)
    std_test_r2 = np.std(test_r2_scores)
    avg_test_mae = np.mean(test_mae_scores)
    std_test_mae = np.std(test_mae_scores)

    return {
        'train_r2': (avg_train_r2, std_train_r2),
        'train_mae': (avg_train_mae, std_train_mae),
        'test_r2': (avg_test_r2, std_test_r2),
        'test_mae': (avg_test_mae, std_test_mae),
    }



# Example usage if this file is executed directly
if __name__ == "__main__":
    _x = [[1, 0], [2, 0], [1, 0], [2, 0], [0, 1], [0, 2]]
    _y = [10, 20, 15, 25, 5, 30]
    groups = [1, 1, 2, 2, 3, 3]

    # Create a GroupedData instance
    data = GroupedData(_x, _y, groups)

    # Split the data with a fixed random state
    train_data, test_data = group_based_train_test_split(data, test_size=0.33, random_state=42)

    print("train data:", train_data)
    print("test data:", test_data)
