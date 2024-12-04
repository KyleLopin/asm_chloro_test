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


def create_sample_grouped_data() -> global_classes.GroupedData:
    """
    Create a sample GroupedData instance for testing purposes.

    Returns:
    - GroupedData: A GroupedData instance containing sample data with features, target values, and group labels.
    """
    # Sample data
    x = [[1, 0],
         [2, 0],
         [1, 0],
         [2, 0],
         [0, 1],
         [0, 2]]
    y = [10, 20, 15, 25, 5, 30]
    groups = [1, 1, 2, 2, 3, 3]

    # Return the data as a GroupedData instance
    return global_classes.GroupedData(x, y, groups)


class TestGroupBasedTrainTestSplit(unittest.TestCase):

    def setUp(self):
        # Use the helper function to create a sample GroupedData instance
        self.grouped_data = create_sample_grouped_data()

    def test_single_split(self):
        """Test group_based_train_test_split with n_splits=1."""
        train_data, test_data = helper_functions.group_based_train_test_split(
            self.grouped_data, test_size=0.3, random_state=42, n_splits=1)

        total_samples = len(train_data.x) + len(test_data.x)
        self.assertEqual(total_samples, len(self.grouped_data.x))

        shared_groups = set(train_data.group).intersection(set(test_data.group))
        self.assertEqual(len(shared_groups), 0)

    def test_multiple_splits(self):
        """Test group_based_train_test_split with n_splits=2."""
        splits = list(
            helper_functions.group_based_train_test_split(
                self.grouped_data, test_size=0.3, random_state=42,
                n_splits=2))

        self.assertEqual(len(splits), 2)

        for train_data, test_data in splits:
            total_samples = len(train_data.x) + len(test_data.x)
            self.assertEqual(total_samples, len(self.grouped_data.x))

            shared_groups = set(train_data.group).intersection(set(test_data.group))
            self.assertEqual(len(shared_groups), 0)


if __name__ == '__main__':
    unittest.main()
