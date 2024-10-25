# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>
"""
This module contains two classes: CVScores and CustomDict.
CVScores is used to keep track of sets of test and training scores for cross-validation.
CustomDict is a custom dictionary class that inherits from the built-in dict class and allows for appending values to lists stored as values.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from dataclasses import dataclass, field

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")


@dataclass
class CVScores:
    """
    Class to keep track of sets of test and training scores for cross validation

    Attributes:
        x (list): List of x-axis values.
        test_means (list): List of mean test scores.
        test_stds (list): List of standard deviations of test scores.
        training_means (list): List of mean training scores.
        training_std (list): List of standard deviations of training scores.
        test_color (str): Color for plotting test data.
        train_color (str): Color for plotting training data.

    Examples:
        >>> cv_scores = CVScores()
        >>> cv_scores.add_scores(0.1, {"test_score": np.array([0.8, 0.82, 0.81]), "train_score": np.array([0.9, 0.88, 0.89])})
        >>> cv_scores.add_scores(0.2, {"test_score": np.array([0.85, 0.86, 0.84]), "train_score": np.array([0.92, 0.91, 0.93])})
        >>> cv_scores.plot()
    """
    x: list = field(default_factory=list)
    test_means: list = field(default_factory=list)
    test_stds: list = field(default_factory=list)
    training_means: list = field(default_factory=list)
    training_std: list = field(default_factory=list)
    test_color: str = "goldenrod"
    train_color: str = "blue"

    def add_scores(self, x: float, scores: dict):
        """
        Add a scores dictionary from cross_validate to the class

        Args:
            x (float): x-axis value for the scores to be plotted with
            scores (dict): dictionary of scores, such as the one cross_validate returns

        Returns:
            None
        """
        self.x.append(x)
        self.test_means.append(np.mean(scores["test_score"]))
        self.test_stds.append(np.std(scores["test_score"]))
        self.training_means.append(np.mean(scores["train_score"]))
        self.training_std.append(np.std(scores["train_score"]))

    def plot(self):
        """
        Plot the test and training scores with error bars.

        Returns:
            None
        """
        plt.plot(self.x, self.test_means,
                 color=self.test_color,
                 label="Test data")
        plt.fill_between(self.x,
                         np.subtract(self.test_means, self.test_stds),
                         np.add(self.test_means, self.test_stds),
                         color=self.test_color, alpha=0.4)
        plt.plot(self.x, self.training_means,
                 color=self.train_color,
                 label="Training data")
        plt.fill_between(self.x,
                         np.subtract(self.training_means, self.training_std),
                         np.add(self.training_means, self.training_std),
                         color=self.train_color, alpha=0.5)
        plt.legend()
        # plt.ylim([0.8, 1.0])


class CustomDict(dict):
    """
    A custom dictionary class that inherits from the built-in dict class and allows for appending values to lists stored as values.

    Methods:
        add_data(input_dict): Adds data from another dictionary, appending values to lists if the key already exists.

    Examples:
        >>> custom_dict = CustomDict({'a': 1, 'b': 2})
        >>> custom_dict.add_data({'b': 3, 'c': 4})
        >>> custom_dict
        {'a': [1], 'b': [2, 3], 'c': [4]}
        >>> custom_dict.add_data({'a': 5, 'c': 6})
        >>> custom_dict
        {'a': [1, 5], 'b': [2, 3], 'c': [4, 6]}
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the CustomDict instance.

        Args:
            *args: Variable length argument list to be passed to the parent dict class.
            **kwargs: Arbitrary keyword arguments to be passed to the parent dict class.
        """
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = [value] if not isinstance(value, list) else value

    def add_data(self, input_dict):
        """
        Add data from another dictionary, appending values to lists if the key already exists.

        Args:
            input_dict (dict): Dictionary containing data to add.

        Returns:
            None
        """
        for key, value in input_dict.items():
            if key in self:
                self[key].append(value)
            else:
                self[key] = [value]


class GroupedData:
    def __init__(self, x, y, group):
        # Convert x, y, and group to pandas DataFrames/Series for consistency
        self.x = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
        self.y = pd.Series(y) if not isinstance(y, pd.Series) else y
        self.group = pd.Series(group) if not isinstance(group, pd.Series) else group

    def __repr__(self):
        return f"GroupedData(x.shape={self.x.shape}, y.shape={self.y.shape}, group.shape={self.group.shape})"
