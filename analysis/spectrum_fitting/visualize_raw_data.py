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
print(plt.style.available)
plt.style.use('seaborn-v0_8-dark')
COLOR_MAP = 'Greens'  # options Greens, YlGn


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
    color_map = mpl.colormaps[COLOR_MAP]
    color_map = mpl.colors.LinearSegmentedColormap.from_list("", ["palegreen", "darkgreen"])

    colormap_min = data["Avg Total Chlorophyll (µg/cm2)"].min()
    colormap_max = data["Avg Total Chlorophyll (µg/cm2)"].max()
    map_norm = mpl.colors.Normalize(vmin=colormap_min,
                                    vmax=colormap_max)
    print(color_map)

    lines = plt.plot(x_data.T)
    # set the color of each line according to its chlorophyll level
    for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
        line.set_color(color_map(map_norm(data["Avg Total Chlorophyll (µg/cm2)"]))[i])
    # scalar_map.set_clim([real_colormap_min, colormap_max])
    # add color bar to show chlorophyll level colors
    plt.colorbar(mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map),
                 orientation="vertical", label="Total Chlorophyll (µg/cm2)",
                 fraction=0.08)
    plt.show()


if __name__ == '__main__':
    visualize_raw_data()
