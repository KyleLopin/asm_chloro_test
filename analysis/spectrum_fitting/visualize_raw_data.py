# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make functions to visualize data from as7262, as7263 and as7265x color sensor data
for the chlorophyll data
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    int_time = 100
    current = "25 mA"
    data = data[data["integration time"] == int_time]
    data = data[data["led current"] == current]
    print(data)
    x_columns = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)
    x_data = data[x_columns]
    # x_data = x_data.mean()
    # make color map
    color_map = mpl.colormaps['YlGn'](np.linspace(data["Avg Total Chlorophyll (µg/cm2)"].min(),
                                                  data["Avg Total Chlorophyll (µg/cm2)"].max(),
                                                  data.shape[0]))
    print(color_map)

    plt.plot(x_data.T)
    plt.show()


if __name__ == '__main__':
    visualize_raw_data()
