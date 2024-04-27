# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittest for functions in the analysis/spectrum_fitting/preprocessors.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
from analysis.spectrum_fitting import preprocessors
import context


class TestPolynomialExpansion(unittest.TestCase):
    """ Test the polynomial_expansion function works """
    def test_simple(self):
        """ Simple test on a small DataFrame """
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["450 nm", "500 nm"])
        results = preprocessors.polynomial_expansion(df)
        correct_df = pd.DataFrame([[1, 2, 1, 4], [3, 4, 9, 16]],
                                  columns=["450 nm", "500 nm", "(450 nm)^2", "(500 nm)^2"])
        assert results.equals(correct_df)

    def test_error(self):
        """ Test putting in an incorrect degree value raises an error """
        with self.assertRaises(ValueError):
            preprocessors.polynomial_expansion(1, degree=1)

    def test_degree_3(self):
        """ Test that the function works if calculating a 3rd degree polynomial expansion """
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["450 nm", "500 nm"])
        results = preprocessors.polynomial_expansion(df, degree=3)
        print(results)
        correct_df = pd.DataFrame([[1, 2, 1, 4, 1, 8], [3, 4, 9, 16, 27, 64]],
                                  columns=["450 nm", "500 nm", "(450 nm)^2", "(500 nm)^2",
                                           "(450 nm)^3", "(500 nm)^3"])
        assert results.equals(correct_df)

    def test_scalar(self):
        """ Test the StandardScalar option works """
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["450 nm", "500 nm"])
        results = preprocessors.polynomial_expansion(df, degree=2,
                                                     standerdize=True)
        print(results)
