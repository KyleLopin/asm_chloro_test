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


class TestFixData(unittest.TestCase):
    def test_add_leave_avgs_simple(self):
        """ Test very simple protocol for fix_outliers.add_leave_average """
        pass


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
