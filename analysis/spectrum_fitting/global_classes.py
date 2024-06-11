# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from dataclasses import dataclass, field

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")


@dataclass
class CVScores:
    """ Class to keep track of sets of test and training scores for cross validation """
    x: list = field(default_factory=list)
    test_means: list = field(default_factory=list)
    test_stds: list = field(default_factory=list)
    training_means: list = field(default_factory=list)
    training_std: list = field(default_factory=list)
    test_color: str = "goldenrod"
    train_color: str = "blue"

    def add_scores(self, x: float, scores: dict):
        """ Add a scores dictionary from cross_validate to the class

        Args:
            x (float): x-axis value for the scores to be plotted with
            scores (dict): dictionary of scores, such as the one cross_validate returns

        Returns:

        """
        self.x.append(x)
        self.test_means.append(scores["test_score"].mean())
        self.test_stds.append(scores["test_score"].std())
        self.training_means.append(scores["train_score"].mean())
        self.training_std.append(scores["train_score"].std())

    def plot(self):
        plt.plot(self.x, self.test_means,
                 color=self.test_color,
                 label="Test data")
        plt.fill_between(self.x,
                         np.subtract(self.test_means, self.test_stds),
                         np.add(self.test_means, self.test_stds),
                         color=self.test_color, alpha=0.4)
        plt.plot(self.x, self.training_means,
                 label="Training data")
        plt.fill_between(self.x,
                         np.subtract(self.training_means, self.training_std),
                         np.add(self.training_means, self.training_std),
                         color=self.train_color, alpha=0.5)
        plt.legend()
        # plt.ylim([0.8, 1.0])
