# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Test functions in the analysis.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import sys

import unittest

# installed libraries
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# local files
import context  # to append sys.path
from analysis.spectrum_fitting import remove_outliers


class TestCalculateSpectrumResidue(unittest.TestCase):
    """Test case for the calculate_spectrum_residue function."""

    def test_simple(self):
        """Test the function with a simple dataset."""
        x = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]], columns=["450 nm", "550 nm"])
        groups = pd.Series([0, 0, 0])
        correct_df = pd.DataFrame([[-1.0, -1.0], [-1.0, -1.0], [2.0, 2.0]],
                                  columns=["450 nm", "550 nm"])
        results = remove_outliers.calculate_spectrum_residue(x, groups)
        assert_frame_equal(results, correct_df, check_exact=False,
                           check_dtype=False)

    def test_2_groups(self):
        """Test the function with two groups."""
        x = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0],
                                [0.0, 0.0], [0.0, 0.0], [2.0, 2.0]],
                         columns=["450 nm", "550 nm"])
        groups = pd.Series([0, 0, 0, 1, 1, 1])
        correct_df = pd.DataFrame([[-1.0, -1.0], [-1.0, -1.0], [2.0, 2.0],
                                         [-1.0, -1.0], [-1.0, -1.0], [2.0, 2.0]],
                                  columns=["450 nm", "550 nm"])
        results = remove_outliers.calculate_spectrum_residue(x, groups)
        assert_frame_equal(results, correct_df, check_exact=False,
                           check_dtype=False)
