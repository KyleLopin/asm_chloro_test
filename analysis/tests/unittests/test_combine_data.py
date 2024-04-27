# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Test the individual functions in the analysis.combine_chloro_spectra_data.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
import context
from analysis.combine_chloro_spectra_data import *


class TestCombineChloroSpectraData(unittest.TestCase):
    """ Tests to check the analysis.combine_chloro_spectra_data.py
     add_chloro_to_df function behaves as expected"""
    def test_combine_data_simple(self):
        """ Simple test """
        chloro_df = pd.DataFrame({
            'Leaf No.': [1, 2, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30]
        })
        chloro_df.set_index('Leaf No.', inplace=True)

        a_df = pd.DataFrame({
            "Leaf No.": [1, 2, 3],
            "foobar": [5, 6, 7]
        })
        a_df.set_index('Leaf No.', inplace=True)

        result_df = add_chloro_to_df(chloro_df, a_df)
        correct_df = pd.DataFrame({
            "foobar": [5, 6, 7],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30]
        }, index=[1, 2, 3])
        correct_df.index.name = "Leaf No."
        assert result_df.equals(correct_df)

    def test_multi_index(self):
        """ Check if multiple samples are included the
         chlorophyll values go to each Leaf number"""
        chloro_df = pd.DataFrame({
            'Leaf No.': [1, 2, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30]
        })
        chloro_df.set_index('Leaf No.', inplace=True)

        b_df = a_df = pd.DataFrame({
            "Leaf No.": [1, 1, 2, 2, 3, 3],
            "foobar": [5, 6, 7, 8, 9, 10]
        })
        b_df.set_index('Leaf No.', inplace=True)

        correct_df = pd.DataFrame({
            "foobar": [5, 6, 7, 8, 9, 10],
            'Total Chlorophyll (µg/cm2)': [10, 10, 20, 20, 30, 30]
        }, index=[1, 1, 2, 2, 3, 3])
        correct_df.index.name = "Leaf No."
        results_df = add_chloro_to_df(chloro_df, b_df)
        pd.testing.assert_frame_equal(results_df, correct_df)

    def test_add_2_columns(self):
        """ Check it adds multiple columns """
        a_df = pd.DataFrame({
            'Leaf No.': [1, 2, 3],
            'add_a': [10, 20, 30],
            'add_b': [11, 21, 31]
        })
        a_df.set_index('Leaf No.', inplace=True)
        b_df = pd.DataFrame({
            "Leaf No.": [1, 1, 2, 2, 3, 3],
            "foobar": [5, 6, 7, 8, 9, 10]
        })
        b_df.set_index('Leaf No.', inplace=True)

        correct_df = pd.DataFrame({
            "foobar": [5, 6, 7, 8, 9, 10],
            'add_a': [10, 10, 20, 20, 30, 30],
            "add_b": [11, 11, 21, 21, 31, 31]
        }, index=[1, 1, 2, 2, 3, 3])
        correct_df.index.name = "Leaf No."
        results_df = add_chloro_to_df(a_df, b_df)
        pd.testing.assert_frame_equal(results_df, correct_df)
