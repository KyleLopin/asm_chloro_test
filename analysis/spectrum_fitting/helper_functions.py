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
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
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


def group_based_train_test_split(data: GroupedData, test_size=0.2, random_state=None) -> tuple[
    GroupedData, GroupedData]:
    """
    Splits the input GroupedData into training and testing sets based on groups,
    ensuring that the same group is not shared between the training and testing sets.

    Args:
    - data (GroupedData): The data to be split, containing features, target, and group information.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int, optional): Controls the randomness of the split for reproducibility.

    Returns:
    - tuple[GroupedData, GroupedData]: The training and testing data, each as a GroupedData instance.
    """
    # Initialize GroupShuffleSplit with specified test_size and random_state
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)

    # Perform the split
    train_idx, test_idx = next(splitter.split(data.x, data.y, groups=data.group))

    # Split the data into training and testing sets
    x_train, x_test = data.x.iloc[train_idx], data.x.iloc[test_idx]
    y_train, y_test = data.y.iloc[train_idx], data.y.iloc[test_idx]
    group_train, group_test = data.group.iloc[train_idx], data.group.iloc[test_idx]

    # Create GroupedData instances for training and testing sets
    train_data = GroupedData(x_train, y_train, group_train)
    test_data = GroupedData(x_test, y_test, group_test)

    return train_data, test_data


def predict_with_average(x, y, groups, test_size=0.2, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    print(x_train, x_test)
    print(y_train, y_test)


# Example usage if this file is executed directly
if __name__ == "__main__":
    x = [[1, 0], [2, 0], [1, 0], [2, 0], [0, 1], [0, 2]]
    y = [10, 20, 15, 25, 5, 30]
    groups = [1, 1, 2, 2, 3, 3]

    # Create a GroupedData instance
    data = GroupedData(x, y, groups)

    # Split the data with a fixed random state
    train_data, test_data = group_based_train_test_split(data, test_size=0.33, random_state=42)

    print("train data:", train_data)
    print("test data:", test_data)
