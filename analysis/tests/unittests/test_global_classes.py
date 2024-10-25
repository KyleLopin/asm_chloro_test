# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import numpy as np
import pandas as pd

# local files
import context  # to append sys.path
from analysis.spectrum_fitting import global_classes


class TestGroupedData(unittest.TestCase):
    def test_grouped_data_initialization_with_pd(self):
        """
        Test GroupedData initialization with pandas DataFrame and Series inputs.
        """
        # Create sample data as pandas DataFrame and Series
        x = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([10, 20, 30])
        group = pd.Series([1, 1, 2])

        # Create an instance of GroupedData
        grouped_data = global_classes.GroupedData(x, y, group)

        # Assertions to check if the attributes are set correctly
        pd.testing.assert_frame_equal(grouped_data.x, x)
        pd.testing.assert_series_equal(grouped_data.y, y)
        pd.testing.assert_series_equal(grouped_data.group, group)

    def test_grouped_data_initialization_with_lists(self):
        """
        Test GroupedData initialization with list inputs.
        """
        # Create sample data as lists
        x = [[1, 2], [3, 4], [5, 6]]
        y = [10, 20, 30]
        group = [1, 1, 2]

        # Create an instance of GroupedData
        grouped_data = global_classes.GroupedData(x, y, group)

        # Expected outputs converted to DataFrame and Series
        expected_x = pd.DataFrame(x)
        expected_y = pd.Series(y)
        expected_group = pd.Series(group)

        # Assertions to check if the attributes are correctly converted
        pd.testing.assert_frame_equal(grouped_data.x, expected_x)
        pd.testing.assert_series_equal(grouped_data.y, expected_y)
        pd.testing.assert_series_equal(grouped_data.group, expected_group)

    def test_grouped_data_initialization_with_np_arrays(self):
        """
        Test GroupedData initialization with numpy array inputs.
        """
        # Create sample data as numpy arrays
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([10, 20, 30])
        group = np.array([1, 1, 2])

        # Create an instance of GroupedData
        grouped_data = global_classes.GroupedData(x, y, group)

        # Expected outputs converted to DataFrame and Series
        expected_x = pd.DataFrame(x)
        expected_y = pd.Series(y)
        expected_group = pd.Series(group)

        # Assertions to check if the attributes are correctly converted
        pd.testing.assert_frame_equal(grouped_data.x, expected_x)
        pd.testing.assert_series_equal(grouped_data.y, expected_y)
        pd.testing.assert_series_equal(grouped_data.group, expected_group)
