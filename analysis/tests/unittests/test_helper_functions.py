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
        results_df = helper_functions.filter_df(selected_row, df2)
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


if __name__ == '__main__':
    unittest.main()

