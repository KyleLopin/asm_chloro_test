# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
stratified_group_shuffle_split.py

This module provides a custom cross-validator class `StratifiedGroupShuffleSplit`
for stratified sampling in regression tasks with grouped data. The class is designed
to act as a drop-in replacement for scikit-learn `GroupShuffleSplit` but adds support
for quantile-based binning of the target variable, allowing stratified splits for regression tasks.

Example usage:
    from stratified_group_shuffle_split import StratifiedGroupShuffleSplit
    splitter = StratifiedGroupShuffleSplit(n_splits=5, test_size=0.2, n_bins=5)
    for train_idx, test_idx in splitter.split(X, y, groups):
        # Train/test split your data based on the indices
"""

from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd


class StratifiedGroupShuffleSplit(BaseCrossValidator):
    """
    StratifiedGroupShuffleSplit cross-validator

    This class provides stratified sampling for regression tasks with grouped data.
    It uses quantile-based binning on the target variable, `y`, allowing the user
    to stratify based on target values while maintaining group integrity across
    train/test splits. It supports group-based stratification with an adjustable
    number of bins for stratification.

    Parameters
    ----------
    n_splits : int, default=1
        Number of re-shuffling & splitting iterations.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    train_size : float, optional
        Proportion of the dataset to include in the train split.
    random_state : int, RandomState instance, or None, optional
        Random state for reproducibility of splits.
    n_bins : int, default=5
        Number of quantile bins to split the target `y` for stratification.
    """

    def __init__(self, n_splits=1, test_size=0.2, train_size=None, random_state=None, n_bins=5):
        # Validate n_bins
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")

        # Validate test_size
        if not (0 < test_size <= 1):
            raise ValueError("test_size must be a float between 0 and 1 (exclusive 0).")

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.n_bins = n_bins

    def split(self, x, y=None, groups=None):
        """
        Generate indices to split data into training and test sets, using quantile-based
        binning of the group means for stratification.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target variable for regression tasks. Will be used to calculate group means for stratification.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used to split the data.

        Yields
        ------
        train_idx : ndarray
            The training set indices for the current split.
        test_idx : ndarray
            The testing set indices for the current split.

        Raises
        ------
        ValueError
            If groups are not provided.
        """
        if y is None:
            raise ValueError("The target 'y' must be provided for stratification.")
        if groups is None:
            raise ValueError("Groups are required for StratifiedGroupShuffleSplit")

        # Create a DataFrame to compute group means
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
        data = pd.DataFrame({'y': y, 'group': groups})
        group_means = data.groupby('group')['y'].mean()  # Calculate mean y per group

        # Bin the group means into quantiles
        group_means_binned = pd.qcut(group_means, q=self.n_bins, labels=False, duplicates='drop')

        # Map each group to its bin
        group_bin_map = dict(zip(group_means.index, group_means_binned))

        # Add a 'y_binned' column to the data based on group mean
        # binning to stratify groups by their mean.
        data['y_binned'] = data['group'].map(group_bin_map)

        # Initialize a random number generator with the specified random state for reproducibility
        rng = np.random.default_rng(self.random_state)

        # Generate the specified number of train-test splits
        for _ in range(self.n_splits):
            # Initialize sets for unique train and test groups
            train_groups, test_groups = set(), set()

            # Stratify based on binned group means
            for bin_value in data['y_binned'].unique():
                # Get unique groups within the current bin
                bin_groups = data[data['y_binned'] == bin_value]['group'].unique()
                # Shuffle groups within the bin to randomize selection
                rng.shuffle(bin_groups)

                # Calculate the number of groups to assign to the test set based on `test_size`
                n_test = int(len(bin_groups) * self.test_size)
                # Assign a subset to test_groups and the rest to train_groups
                test_groups.update(bin_groups[:n_test])
                train_groups.update(bin_groups[n_test:])

            # Ensure train and test groups are mutually exclusive
            test_groups -= train_groups

            # Convert groups to Boolean indices for train/test splits
            train_idx = data['group'].isin(train_groups).values  # True if group is in train_groups
            test_idx = data['group'].isin(test_groups).values
            # Yield the indices of train and test samples as arrays of indices
            yield np.where(train_idx)[0], np.where(test_idx)[0]

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits, which is specified during initialization.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        int
            The number of re-shuffling & splitting iterations.
        """
        return self.n_splits
