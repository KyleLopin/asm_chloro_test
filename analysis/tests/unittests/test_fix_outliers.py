# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Test the individual functions in the analysis.chlorophyll.fix_outliers.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
import analysis.chlorophyll_measurements.fix_outliers as fix_outliers


class TestAddLeaveAverages(unittest.TestCase):
    """ Tests the funtion analysis.chlorophyll_measurements.fix_outliers add_leave_averages """
    def test_add_leave_avgs_simple(self):
        """ Test very simple protocol for fix_outliers.add_leave_average """
        # Create a dataframe with multiple Leaf No. and column_to_average columns
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 1],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30]
        })
        # create the correct dataframe to compare to
        df_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 1],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30],
            'Avg Total Chlorophyll (µg/cm2)':
                [20.0, 20.0, 20.0]
        })

        # Call the add_leave_average function
        result = fix_outliers.add_leave_averages(df, 'Total Chlorophyll (µg/cm2)')

        # Assert that the result is correct
        assert result.equals(df_correct)

    def test_add_leave_avgs_default_args(self):
        """ Test that the add_leave_args will work with default arguments """
        # Create a dataframe with multiple Leaf No. and column_to_average columns
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60]
        })
        # create the correct dataframe to compare to
        df_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60],
            'Avg Total Chlorophyll (µg/cm2)':
                [15.0, 15.0, 35.0, 35.0, 55.0, 55.0]
        })

        # Call the add_leave_average function
        result = fix_outliers.add_leave_averages(df)

        # Assert that the result is correct
        assert result.equals(df_correct)

    def test_add_leave_avgs_with_multiple_leaf_nums_and_column_to_average(self):
        """ Test typical use case for fix_outliers.add_leave_average"""
        # Create a dataframe with multiple Leaf No. and column_to_average columns
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60]
        })
        # create the correct dataframe to compare to
        df_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60],
            'Avg Total Chlorophyll (µg/cm2)':
                [15.0, 15.0, 35.0, 35.0, 55.0, 55.0]
        })

        # Call the add_leave_average function
        result = fix_outliers.add_leave_averages(df, 'Total Chlorophyll (µg/cm2)')

        # Assert that the result is correct
        assert result.equals(df_correct)

    def test_raise_key_error_no_leaf_num(self):
        """ Test that the function raises a KeyError if the dataframe is missing the
        correct column_values_to_average """
        df = pd.DataFrame({'Leaf No.': [1, 1, 1], 'Column1': [1, 2, 3], 'Column2': [4, 5, 6]})
        # with unittest.raises(KeyError):
        #     fix_outliers.add_leave_average(df, 'Total Chlorophyll (µg/cm2)')
        with self.assertRaises(KeyError) as context:
            fix_outliers.add_leave_averages(df, 'Total Chlorophyll (µg/cm2)')

    def test_raise_wrong_groupby(self):
        """ Test that passing in a data frame with a wrong column_to_groupby raises an error """
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60]
        })
        with self.assertRaises(KeyError) as context:
            fix_outliers.add_leave_averages(df, column_to_groupby="foobar")

    def test_generalize_groupby(self):
        """ Test that the columns_to_groupby variable works"""
        # Create a dataframe with multiple Leaf No. and column_to_average columns
        df = pd.DataFrame({
            'Car': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60]
        })
        # create the correct dataframe to compare to
        df_correct = pd.DataFrame({
            'Car': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60],
            'Avg Total Chlorophyll (µg/cm2)':
                [15.0, 15.0, 35.0, 35.0, 55.0, 55.0]
        })

        # Call the add_leave_average function
        result = fix_outliers.add_leave_averages(df, column_to_groupby="Car")

        # Assert that the result is correct
        assert result.equals(df_correct)

    def test_generalize_column_to_average(self):
        """ Test you can set hte column_values_to_average can be set """
        # Create a dataframe with multiple Leaf No. and column_to_average columns
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Foobar': [10, 20, 30, 40, 50, 60]
        })
        # create the correct dataframe to compare to
        df_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Foobar': [10, 20, 30, 40, 50, 60],
            'Avg Foobar':
                [15.0, 15.0, 35.0, 35.0, 55.0, 55.0]
        })

        # Call the add_leave_average function
        result = fix_outliers.add_leave_averages(df, column_values_to_average="Foobar")

        # Assert that the result is correct
        assert result.equals(df_correct)
