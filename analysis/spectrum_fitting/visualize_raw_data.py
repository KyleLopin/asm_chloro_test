# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make functions to visualize data from as7262, as7263 and as7265x color sensor data
for the chlorophyll data
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import matplotlib.pyplot as plt

# local files
import get_data


def visualize_raw_data(sensor="as7262", leaf="banana",
                       measurement_type="reflectance"):
    """ Make a graph to visualize basic data, use this to test what is best for paper


    Args:
        sensor:
        leaf:
        measurement_type:

    Returns:

    """
    data = get_data.get_data(sensor=sensor, leaf=leaf,
                             measurement_type=measurement_type,
                             mean=True)
    print(data)


if __name__ == '__main__':
    visualize_raw_data()
