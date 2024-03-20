# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittests for the get_data.py file in analysis/spectrum_fitting directory
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import unittest

# installed libraries
import numpy as np
import pandas as pd

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
        for _type in ["raw", "reflectance"]:
            for sensor in ["as7262", "as7263"]:
                for leaf in ALL_LEAVES:
                    data = get_data.get_data(sensor=sensor, leaf=leaf,
                                             measurement_type=_type)
                    print(f"({sensor}, {leaf}, {_type}): {data.shape}")


class TestGetXY(unittest.TestCase):
    """ Test the get_x_y function """
    def test_shapes(self):
        for _type in ["raw", "reflectance"]:
            for sensor in ["as7262", "as7263"]:
                for leaf in ALL_LEAVES:
                    int_times, leds, currents = get_data.get_options(
                        sensor=sensor, leaf=leaf,measurement_type=_type)
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
