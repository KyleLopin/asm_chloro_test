# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittests for the get_data.py file in analysis/spectrum_fitting directory
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import pandas as pd

# local files
from analysis.spectrum_fitting import get_data

ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
ALL_SENSORS = ["as7265x", "as7262", "as7263"]


class TestGetOptions(unittest.TestCase):
    """ Test the get_options function """
    def test_get_options_as7262_3(self):
        """ Test all the options for the as7262 and as7263 for every combination is correct"""
        for _type in ["raw", "reflectance"]:
            for sensor in ["as7262", "as7263"]:
                for leaf in ALL_LEAVES:
                    int_times, leds, currents =  get_data.get_options(
                        sensor=sensor, leaf=leaf, measurement_type=_type)
                    self.assertListEqual(int_times, [50, 100, 150, 200, 250])
                    self.assertListEqual(leds, ["White LED"])
                    self.assertListEqual(currents, ['12.5 mA', '25 mA', '50 mA', '100 mA'])

    def test_get_options_as7265x(self):
        """ Test all the options for the as7265x for every combination is correct"""
        for _type in ["raw", "reflectance"]:
            for leaf in ALL_LEAVES:
                int_times, leds, currents = get_data.get_options(
                    sensor="as7265x", leaf=leaf, measurement_type=_type)
                self.assertListEqual(int_times, [50, 100, 150])
                self.assertListEqual(leds, ["b'White'", "b'IR'", "b'UV'", "b'White IR'",
                                            "b'White UV'", "b'UV IR'", "b'White UV IR'"])
                self.assertListEqual(currents, ['12.5 mA', '25 mA', '50 mA', '100 mA'])


class TestGetData(unittest.TestCase):
    """ Test the get_data function"""
    def test_all_combination_size(self):
        """ Test that the get_data.get_data function returns data of the correct shape """
        for _type in ["raw", "reflectance"]:
            for sensor in ["as7262", "as7263"]:
                for leaf in ALL_LEAVES:
                    data = get_data.get_data(sensor=sensor, leaf=leaf,
                                             measurement_mode=_type)
                    self.assertTrue(data.shape[0] >= 5999)
                    self.assertTrue(data.shape[1] <= 20)


class TestGetXY(unittest.TestCase):
    """ Test the get_x_y function """
    def test_shapes(self):
        """ Test that the get_data.get_x_y function returns the correct shape data, uses the
            shape_test method to reduce nesting """
        for _type in ["raw", "reflectance"]:
            for sensor in ["as7262", "as7263"]:
                for leaf in ALL_LEAVES:
                    self.shape_test(sensor, leaf, _type)

    def shape_test(self, sensor, leaf, _type):
        """ helper function for the main method test_shapes """
        int_times, leds, currents = get_data.get_options(
            sensor=sensor, leaf=leaf, measurement_type=_type)
        for int_time in int_times:
            for led in leds:
                for current in currents:
                    x, y = get_data.get_x_y(
                        sensor=sensor, leaf=leaf, measurement_type=_type,
                        int_time=int_time, led=led, led_current=current)
                    self.assertTrue(x.shape[0] >= 299)
                    self.assertTrue(x.shape[1] == 6)
                    self.assertTrue(y.shape[0] >= 299)
                    self.assertTrue(y.shape[1] == 6)


class TestGetDataSlices(unittest.TestCase):
    """ Test the get_data.get_data_slices function """
    def test_simple(self):
        """ Simple test for development """
        df = pd.DataFrame({
            'Leaf No.': [1, 2, 3],
            'foo': [10, 20, 30],
            'current': ["1 mA", "2 mA", "3 mA"]
        })
        results = get_data.get_data_slices(df, selected_column="current",
                                           values=["1 mA", "2 mA"])
        correct_df = df = pd.DataFrame({
            'Leaf No.': [1, 2],
            'foo': [10, 20],
            'current': ["1 mA", "2 mA"]
        })
        assert correct_df.equals(results)

    def test_real_data(self):
        """ Test that getting the real data returns the correct results by going through each
         measurement type, sensor and leaf combination, uses the helper method each_condition
         to do the actual asserting """
        for _type in ["raw", "reflectance"]:
            for sensor in ["as7262", "as7263", "as7265x"]:
                for leaf in ALL_LEAVES:
                    data = get_data.get_data(sensor=sensor, leaf=leaf,
                                             measurement_mode=_type)
                    self.each_condition(data, leaf, sensor)

    def each_condition(self, data, leaf, sensor):
        """ Take a DataFrame and go through each led, integration time and led current
         and assert it has the correct shape """
        for led in data["led"].unique():
            for int_time in data["integration time"].unique():
                for current in data["led current"].unique():
                    # print(led, int_time, current)
                    result = get_data.get_data_slices(data, "led", [led])
                    result = get_data.get_data_slices(result, "integration time",
                                                      [int_time])
                    result = get_data.get_data_slices(result, "led current",
                                                      [current])
                    # print(result.shape, (led, int_time, current, leaf, sensor))
                    if ((led, int_time, current, leaf, sensor) ==
                            ("White LED", 200, "50 mA", "banana", "as7262")):
                        # this read has 1 that was saturated so is 1 smaller than normal
                        self.assertTrue(result.shape == (299, 20))
                    elif sensor == "as7265x":  # this has a different shape because of more channels
                        if current != "100 mA":  # not all 100 mA conditions were tested so skip it
                            self.assertTrue(result.shape == (300, 32))
                    else:
                        self.assertTrue(result.shape == (300, 20))
