# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd
from pandas.testing import assert_frame_equal

# local files
import context  # to append sys.path
from analysis.spectrum_fitting import helper_functions
from analysis.spectrum_fitting import global_classes


class TestFilteredDF(unittest.TestCase):
    """Test cases for the filter_df function."""

    def test_single_row_match(self):
        df1 = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, 35, 40],
            'Score': [80, 75, 85, 90]
        })

        df2 = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, 35, 40],
            'Score': [85, 70, 80, 95]
        })

        # Select a row from the first DataFrame
        selected_row = df1.iloc[0]

        # Filter df2 based on selected_row
        results_df = helper_functions.filter_df(selected_row, df2)
        correct_df = pd.DataFrame({
            'ID': [1],
            'Name': ['Alice'],
            'Age': [25],
            'Score': [85]
        })
        assert_frame_equal(results_df, correct_df)

    def test_multiple_row_match(self):
        df1 = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Name': ['Alice', 'Bob', 'Alice', 'David'],
            'Age': [25, 30, 25, 40],
            'Score': [80, 75, 85, 90]
        })

        df2 = pd.DataFrame({
            'ID': [1, 2, 1, 4],
            'Name': ['Alice', 'Bob', 'Alice', 'David'],
            'Age': [25, 30, 25, 40],
            'Score': [85, 70, 90, 95]
        })

        # Select a row from the first DataFrame
        selected_row = df1.iloc[0]

        # Filter df2 based on selected_row
        results_df = helper_functions.filter_df(selected_row, df2)
        correct_df = pd.DataFrame({
            'ID': [1, 1],
            'Name': ['Alice', 'Alice'],
            'Age': [25, 25],
            'Score': [85, 90]
        })
        assert_frame_equal(results_df.reset_index(drop=True), correct_df)

    def test_different_named_columns(self):
        df1 = pd.DataFrame({
            'Key': [1, 2, 3, 4],
            'FirstName': ['Alice', 'Bob', 'Charlie', 'David'],
            'Years': [25, 30, 35, 40],
            'Marks': [80, 75, 85, 90]
        })

        df2 = pd.DataFrame({
            'Key': [1, 2, 3, 4],
            'FirstName': ['Alice', 'Bob', 'Charlie', 'David'],
            'Years': [25, 30, 35, 40],
            'Marks': [85, 70, 80, 95]
        })

        # Select a row from the first DataFrame
        selected_row = df1.iloc[0]

        # Filter df2 based on selected_row
        results_df = helper_functions.filter_df(selected_row, df2,
                                                exclude_columns='Marks')
        print(results_df)
        correct_df = pd.DataFrame({
            'Key': [1],
            'FirstName': ['Alice'],
            'Years': [25],
            'Marks': [85]
        })
        assert_frame_equal(results_df, correct_df)

    def test_exclude_columns_argument(self):
        df1 = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, 35, 40],
            'Score': [80, 75, 85, 90],
            'Rank': [1, 2, 3, 4]
        })

        df2 = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, 35, 40],
            'Score': [85, 70, 80, 95],
            'Rank': [1, 2, 3, 4]
        })

        # Select a row from the first DataFrame
        selected_row = df1.iloc[0]

        # Filter df2 based on selected_row excluding 'Rank' column
        results_df = helper_functions.filter_df(selected_row, df2,
                                                exclude_columns=['Score', 'Rank'])
        correct_df = pd.DataFrame({
            'ID': [1],
            'Name': ['Alice'],
            'Age': [25],
            'Score': [85],
            'Rank': [1]
        })
        assert_frame_equal(results_df, correct_df)


class TestGroupBasedTrainTestSplit(unittest.TestCase):
    def test_group_based_train_test_split(self):
        """
        Test group_based_train_test_split to ensure data is split correctly
        based on group membership.
        """
        # Create sample data
        x = [[1, 0],
             [2, 0],
             [1, 0],
             [2, 0],
             [0, 1],
             [0, 2]]
        y = [10, 20, 15, 25, 5, 30]
        groups = [1, 1, 2, 2, 3, 3]

        # Convert data to GroupedData instance
        grouped_data = global_classes.GroupedData(x, y, groups)

        # Perform group-based train-test split with a specified random state for consistency
        train_data, test_data = helper_functions.group_based_train_test_split(
            grouped_data, test_size=0.2, random_state=44
        )

        # Assertions to check if the splits are as expected
        # Use a fixed random state to know the expected results for testing
        expected_train_x = pd.DataFrame([[1, 0], [2, 0], [1, 0], [2, 0]])
        expected_train_y = pd.Series([10, 20, 15, 25])
        expected_train_group = pd.Series([1, 1, 2, 2])

        # can't ignore indexs of dataFrames yet
        expected_test_x = pd.DataFrame([[0, 1], [0, 2]], index=[4, 5])
        expected_test_y = pd.Series([5, 30], index=[4, 5])
        expected_test_group = pd.Series([3, 3], index=[4, 5])

        # Check if the train split matches the expected data
        pd.testing.assert_frame_equal(train_data.x, expected_train_x)
        pd.testing.assert_series_equal(train_data.y, expected_train_y)
        pd.testing.assert_series_equal(train_data.group, expected_train_group)

        # Check if the test split matches the expected data
        pd.testing.assert_frame_equal(test_data.x, expected_test_x)
        pd.testing.assert_series_equal(test_data.y, expected_test_y)
        pd.testing.assert_series_equal(test_data.group, expected_test_group)


if __name__ == '__main__':
    unittest.main()

