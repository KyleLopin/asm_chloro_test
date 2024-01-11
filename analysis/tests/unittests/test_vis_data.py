# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Test the individual functions in the analysis.chlorophyll.vis_data.py file.

Tests the helper functions only
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
import context
from analysis.chlorophyll_measurements import vis_data


class TestGetResidueRange(unittest.TestCase):
    def test_max_limit(self):
        """ Test when the maximum value in the residue range is the limit.
        This will use all leaves and find the max residue after eliminating all
        outliers.
        Removing the nested max used initially but this will make sure it does not
        start giving the wrong answer after refactoring """
        self.assertAlmostEqual(vis_data.get_residue_range(), 11.084588116666)
