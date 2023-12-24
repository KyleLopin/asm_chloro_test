# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Test the individual functions in the analysis.chlorophyll.fix_outliers.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import pathlib
import sys
import unittest

# installed libraries
import numpy as np
import pandas as pd

# local files
# test_file_folder = pathlib.Path().absolute().parent.parent / "chlorophyll_measurements"
# sys.path.append(str(test_file_folder))
import context
from analysis.chlorophyll_measurements import fix_outliers


class TestAddLeaveAverages(unittest.TestCase):
    """ Tests the funtion analysis.chlorophyll_measurements.fix_outliers add_leave_averages """
    def test_simple(self):
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

    def test_default_args(self):
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

    def test_overwrites(self):
        """ Test that when a data frame with an existing average column is added,
        it will overwrite that column with new data """
        df_start = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60],
            'Avg Total Chlorophyll (µg/cm2)':
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        })
        df_end = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60],
            'Avg Total Chlorophyll (µg/cm2)':
                [15.0, 15.0, 35.0, 35.0, 55.0, 55.0]
        })
        result = fix_outliers.add_leave_averages(df_start)

        # Assert that the result is correct
        assert result.equals(df_end)

    def test_multiple_leaf_nums_and_column_to_average(self):
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
        with self.assertRaises(KeyError) as _:
            fix_outliers.add_leave_averages(df, 'Total Chlorophyll (µg/cm2)')

    def test_raise_wrong_groupby(self):
        """ Test that passing in a data frame with a wrong column_to_groupby raises an error """
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 2, 2, 3, 3],
            'Total Chlorophyll (µg/cm2)': [10, 20, 30, 40, 50, 60]
        })
        with self.assertRaises(KeyError) as _:
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


class TestGaussian(unittest.TestCase):
    """ Test the Gaussian works, not robust """
    def test_basic_gaussian(self):
        """ Test the Gaussian works """
        x_values = np.linspace(-5, 5, 11)
        correct_results = [0.00037266531720786707, 0.033546262790251184, 1.1108996538242306,
                           13.53352832366127, 60.653065971263345, 100.0,
                           60.653065971263345, 13.53352832366127, 1.1108996538242306,
                           0.033546262790251184, 0.00037266531720786707]

        results = fix_outliers.gauss_function(x_values, 100, 0, 1)
        self.assertListEqual(results.tolist(), correct_results)


class TestRemoveOutliersRecursive(unittest.TestCase):
    """ test the function remove_outliers_recursive in fix_outliers.py, is same as
    TestRemoveOutliers but just with remove_outliers changed to remove_outliers_recursive"""
    def test_basic_remove_1_outlier(self):
        """ Basic test fix_outliers.add_leave_averages was developed with """
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'Foobar': [0, 1, 2, 1, 2, 3, 3, 3, 120, 0, 1, 2, 0, 1, 2]
        })
        df_final_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            'Foobar': [0, 1, 2, 1, 2, 3, 3, 3, 0, 1, 2, 0, 1, 2],
            "Avg Foobar": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }, index=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
        results, idx_removed = fix_outliers.remove_outliers_recursive(df, column_name="Foobar")
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [8])

    def test_basic_remove_2_outlier(self):
        """ Basic test fix_outliers.add_leave_averages was developed with """
        df = pd.DataFrame({'SampleNumber': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                           'Value': [1, 2, 3, 3, 30, 3, 5, 6, 7, 7, 8, 9, 9, 10, 11]})
        df_final_correct = pd.DataFrame({
            'SampleNumber': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'Value': [1, 2, 3, 3, 3, 5, 6, 7, 7, 8, 9, 9, 10, 11],
            "Avg Value": [2.0, 2.0, 2.0, 3.0, 3.0, 6.0, 6.0, 6.0,
                          8.0, 8.0, 8.0, 10.0, 10.0, 10.0]
        }, index=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        results, idx_removed = fix_outliers.remove_outliers_recursive(
            df, column_sample_number="SampleNumber", column_name="Value")
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [4])

    def test_remove_2_outliers(self):
        """ Test that the fix_outliers.remove_outliers will remove 2 samples correctly"""
        df = pd.DataFrame({'SampleNumber': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                           'Value': [1, 2, 3, 20, 5, 6, 10, 100, 5, 5]})
        df_final_correct = pd.DataFrame({'SampleNumber': [1, 1, 2, 2, 3, 3, 5, 5],
                                         'Value': [1, 2, 3, 20, 5, 6, 5, 5],
                                         "Avg Value": [1.5, 1.5, 11.5, 11.5, 5.5, 5.5, 5.0, 5.0]},
                                        index=[0, 1, 2, 3, 4, 5, 8, 9])
        results, idx_removed = fix_outliers.remove_outliers_recursive(df, column_sample_number="SampleNumber",
                                                                      column_name="Value", sigma_cutoff=2)
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [6, 7])


class TestRemoveOutliers(unittest.TestCase):
    """ test the function remove_outliers in fix_outliers.py, is same as
    TestRemoveOutliersRecursive but just with remove_outliers_recursive changed to remove_outliers"""
    def test_basic_remove_1_outlier(self):
        """ Basic test fix_outliers.add_leave_averages was developed with """
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'Foobar': [0, 1, 2, 1, 2, 3, 3, 3, 120, 0, 1, 2, 0, 1, 2]
        })
        df_final_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            'Foobar': [0, 1, 2, 1, 2, 3, 3, 3, 0, 1, 2, 0, 1, 2],
            "Avg Foobar": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }, index=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
        results, idx_removed = fix_outliers.remove_outliers(df, column_name="Foobar")
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [8])

    def test_basic_remove_2_outlier(self):
        """ Basic test fix_outliers.add_leave_averages was developed with """
        df = pd.DataFrame({'SampleNumber': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                           'Value': [1, 2, 3, 3, 30, 3, 5, 6, 7, 7, 8, 9, 9, 10, 11]})
        df_final_correct = pd.DataFrame({
            'SampleNumber': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'Value': [1, 2, 3, 3, 3, 5, 6, 7, 7, 8, 9, 9, 10, 11],
            "Avg Value": [2.0, 2.0, 2.0, 3.0, 3.0, 6.0, 6.0, 6.0,
                          8.0, 8.0, 8.0, 10.0, 10.0, 10.0]
        }, index=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        results, idx_removed = fix_outliers.remove_outliers(df, column_sample_number="SampleNumber",
                                                            column_name="Value")
        print(results)
        print(idx_removed)

        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [4])

    def test_remove_2_outliers(self):
        """ Test that the fix_outliers.remove_outliers will remove 2 samples correctly"""
        df = pd.DataFrame({'SampleNumber': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                           'Value': [1, 2, 3, 21, 5, 6, 10, 100, 5, 5]})
        df_final_correct = pd.DataFrame({'SampleNumber': [1, 1, 2, 2, 3, 3, 5, 5],
                                         'Value': [1, 2, 3, 21, 5, 6, 5, 5],
                                         "Avg Value": [1.5, 1.5, 12.0, 12.0, 5.5, 5.5, 5.0, 5.0]},
                                        index=[0, 1, 2, 3, 4, 5, 8, 9])
        results, idx_removed = fix_outliers.remove_outliers(df, column_sample_number="SampleNumber",
                                                            column_name="Value", sigma_cutoff=2)
        print(results)
        print(idx_removed)
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [6, 7])


class TestRealDataProblems(unittest.TestCase):
    """ When running the real chlorophyll data, these test are to examine any
        apparent anomalies """
    def test_select_banana_cases(self):
        select_df = pd.DataFrame(
            {'Leaf No.': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4,
                          10: 4, 11: 4, 12: 5, 13: 5, 14: 5, 15: 6, 16: 6, 17: 6,
                          18: 7, 19: 7, 20: 7, 21: 8, 22: 8, 23: 8, 24: 9, 25: 9,
                          26: 9, 75: 26, 76: 26, 77: 26},
             'Spot': {0: 'A', 1: 'B', 2: 'C', 3: 'A', 4: 'B', 5: 'C', 6: 'A', 7: 'B',
                      8: 'C', 9: 'A', 10: 'B', 11: 'C', 12: 'A', 13: 'B', 14: 'C',
                      15: 'A', 16: 'B', 17: 'C', 18: 'A', 19: 'B', 20: 'C', 21: 'A',
                      22: 'B', 23: 'C', 24: 'A', 25: 'B', 26: 'C', 75: 'A', 76: 'B', 77: 'C'},
             'Total Chlorophyll (µg/cm2)':
                 {0: 40.8005989, 1: 36.21977789, 2: 39.28469972,  3: 71.79441654,
                  4: 69.1351489, 5: 73.29814435, 6: 62.05067553, 7: 65.15715263,
                  8: 64.36948433, 9: 64.49128292, 10: 56.98083516, 11: 55.03746198,
                  12: 59.41880173, 13: 62.39419962, 14: 60.6461767, 15: 56.80162222,
                  16: 47.28889675, 17: 48.35210676, 18: 67.44996817, 19: 64.93131027,
                  20: 68.43173233, 21: 82.75260192, 22: 83.41055858, 23: 78.4097569,
                  24: 87.41269954, 25: 70.22698357, 26: 74.62139539, 75: 59.45025111,
                  76: 26.10216217, 77: 62.9294617}})
        _, idx_removed = fix_outliers.remove_outliers_recursive(
            select_df, column_name='Total Chlorophyll (µg/cm2)',
            sigma_cutoff=3)
        self.assertListEqual([76, 24], idx_removed)


class TestDropMeasurement(unittest.TestCase):
    """ test the drop_measurement_w_sample_check function in fix_outliers.py """
    def test_basic_1_sample(self):
        """ test it drops 1 sample when there are 3 samples for the leaf number"""
        df = pd.DataFrame({
            'Leaf No.': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'Foobar': [0, 1, 2, 1, 2, 3, 3, 3, 120, 0, 1, 2, 0, 1, 2]
        })
        df_final_correct = pd.DataFrame({
            'Leaf No.': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            'Foobar': [0, 1, 2, 1, 2, 3, 3, 3, 0, 1, 2, 0, 1, 2],
        }, index=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14])
        results, idx_removed = fix_outliers.drop_measurement_w_sample_check(df, 8)
        print(results)
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [8])

    def test_only_2_samples(self):
        """ test it drops all the Samples when there is only 2 sample numbers left """
        df = pd.DataFrame({'SampleNumber': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                           'Value': [1, 2, 3, 20, 5, 6, 10, 100, 5, 5]})
        df_final_correct = pd.DataFrame({'SampleNumber': [1, 1, 2, 2, 3, 3, 5, 5],
                                         'Value': [1, 2, 3, 20, 5, 6, 5, 5]},
                                        index=[0, 1, 2, 3, 4, 5, 8, 9])
        results, idx_removed = fix_outliers.drop_measurement_w_sample_check(
            df, 7, sample_number_column="SampleNumber")
        print(f"results: {results}")
        print(f"removed indexes: {idx_removed}")
        pd.testing.assert_frame_equal(results, df_final_correct)
        self.assertListEqual(idx_removed, [6, 7])
