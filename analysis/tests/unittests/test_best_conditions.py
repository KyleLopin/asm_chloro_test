# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Test functions in the analysis/spectrum_fitting/best_conditions

Test that the functions make_pg_anova_table and read_conditions_file return the correct
sized dataframe with proper column names, can not test they are correct as they
will be slightly different for different cv measurements.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
import context
from analysis.spectrum_fitting import check_best_conditions


class TestMakePgAnovaTable(unittest.TestCase):
    def test_check_size_and_columns(self):
        for sensor in ["as7262", "as7263"]:
            for _type in ['raw', 'reflectance', 'absorbance']:
                df = check_best_conditions.make_pg_anova_table(
                    sensor=sensor, measurement_type=_type,
                    score_type='r2'
                )
                self.assertTupleEqual(df.shape, (100, 3))
                self.assertListEqual(list(df.columns), ['current', 'int time', 'r2'])


class TestReadConditionFile(unittest.TestCase):
    def test_check_size_columns_and_index(self):
        for sensor in ["as7262", "as7263"]:
            for _type in ['raw', 'reflectance', 'absorbance']:
                df = check_best_conditions.read_conditions_file(
                    sensor=sensor, measurement_type=_type,
                    score_type="r2"
                )
                self.assertTupleEqual(df.shape, (5, 20))
                self.assertListEqual(list(df.index), [0, 1, 2, 3, 4])
