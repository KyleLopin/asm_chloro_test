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


def visualize_raw_data(ax: plt.Axes = None, fig: plt.Figure = None,
                       sensor: str = "as7262", leaf: str = "banana",
                       measurement_type: str = "reflectance"):
    """ Make a graph to visualize basic data, use this to test what is best for paper


    Args:
        sensor:
        leaf:
        measurement_type:

    Returns:

    """
    if not ax:
        fig, ax = plt.subplots(1, 1)
    data = get_data.get_data(sensor=sensor, leaf=leaf,
                             measurement_type=measurement_type,
                             mean=True)
    int_time = 100
    current = "25 mA"
    data = data[data["integration time"] == int_time]
    data = data[data["led current"] == current]
    x_columns = []
    x = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)
            x.append(int(column.split()[0]))
    x_data = data[x_columns]
    print(x)

    # make color map
    # color_map = mpl.colormaps[COLOR_MAP]
    color_map = mpl.colors.LinearSegmentedColormap.from_list("", ["palegreen", "darkgreen"])

    colormap_min = data["Avg Total Chlorophyll (µg/cm2)"].min()
    colormap_max = data["Avg Total Chlorophyll (µg/cm2)"].max()
    map_norm = mpl.colors.Normalize(vmin=colormap_min,
                                    vmax=colormap_max)
    # print(color_map)
    lines = ax.plot(x, x_data.T, alpha=0.7)
    # cax = fig.add_axes([.9, 0.1, 0.05, 0.5])
    # set the color of each line according to its chlorophyll level
    for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
        line.set_color(color_map(map_norm(data["Avg Total Chlorophyll (µg/cm2)"]))[i])
    # make mean line
    x_mean = x_data.mean()
    ax.plot(x, x_mean, color="black", label="mean")
    # scalar_map.set_clim([real_colormap_min, colormap_max])
    # add color bar to show chlorophyll level colors
    # fig.colorbar(mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map),
    #              cax=cax, orientation="vertical",
    #              label="Total Chlorophyll (µg/cm2)", fraction=0.08)



if __name__ == '__main__':
    figure, axs = plt.subplots(2, 2,
                               sharex="col")
    for i, sensor in enumerate(["as7262", "as7263"]):
        print(f"i = {i}")
        visualize_raw_data(ax=axs[0, i], fig=figure, sensor=sensor)
        axs[0, i].set_title(f"{sensor.upper()}\nReflectance")
        axs[0, i].set_ylabel("% reflectance")
        visualize_raw_data(ax=axs[1, i], fig=figure,
                           measurement_type="raw",
                           sensor=sensor)
        axs[1, i].set_title("Raw Measurements")
        axs[1, i].set_xlabel("Wavelength (nm)")
        axs[1, i].set_ylabel("Counts")


    plt.tight_layout()
    plt.show()
