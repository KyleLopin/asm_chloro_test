# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make functions to visualize data from as7262, as7263 and as7265x color sensor data
for the chlorophyll data.
The 2 sensor graph needs better spacing still
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache

# installed libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# local files
import get_data
plt.style.use('seaborn-v0_8-dark')
COLOR_MAP = 'Greens'  # options: Greens, YlGn
COLOR_MAP_FROM_LIST = ["palegreen", "darkgreen"]
INT_TIME = 100  # the integration time to display
CURRENT = "25 mA"  # the current to use in the measurement to display
COLOR_BAR_AXIS = [.90, .1, 0.02, 0.8]
SUBPLOT_RIGHT_PAD = 0.87
WIDTH_PADDING = 0.3


# helper functions
@lru_cache()  # this gets called 5-7 times so just memorize it instead of making a class
def make_color_map(color_min: float, color_max: float
                   ) -> tuple[mpl.colors.LinearSegmentedColormap,
                              mpl.colors.Normalize]:
    """ Make a linear segment color map from a list """
    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "", COLOR_MAP_FROM_LIST)

    map_norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    return color_map, map_norm


def visualize_raw_data(ax: plt.Axes = None, sensor: str = "as7262",
                       leaf: str = "banana", led: str = "",
                       measurement_type: str = "reflectance") -> mpl.cm.ScalarMappable:
    """ Make a graph to visualize basic data, use this to test what is best for paper

    Args:
        ax (plt.Axes): Axis to put the data, if None a new axis and figure will be made
        sensor (str): Which sensor's data to plot, can be "as7262", "as7263", and "as7265x"
        leaf (str): Which leaf's data to plat, works for "mango",
        "banana","jasmine", "rice", "sugarcane"
        measurement_type (str): Which data type to use, "raw" for raw data counts or
        "reflectance" for reflectance values
        led (str): if truthy, select a subset of the data that uses that specified led

    Returns:
        Returns the ScalarMappable to make a color map that corresponds to
        the line color with and adds plot to the given axis.

    """
    if not ax:
        _, ax = plt.subplots(1, 1)
    data = get_data.get_data(sensor=sensor, leaf=leaf,
                             measurement_type=measurement_type,
                             mean=True)
    if led:
        if led not in data["led"].unique():
            raise KeyError(f"The LED: {'led'} not in data, valid LEDs are {data['led'].unique()}")
        data = data[data["led"] == led]
    data = data[data["integration time"] == INT_TIME]
    data = data[data["led current"] == CURRENT]
    x_columns = []
    x = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)
            x.append(int(column.split()[0]))
    x_data = data[x_columns]
    if measurement_type == 'raw':  # make the y labels smaller
        x_data = x_data / 1000
    # make color map
    color_map, map_norm = make_color_map(data["Avg Total Chlorophyll (µg/cm2)"].min(),
                                         data["Avg Total Chlorophyll (µg/cm2)"].max())
    # print(color_map)
    lines = ax.plot(x, x_data.T, alpha=0.7)
    # cax = fig.add_axes([.9, 0.1, 0.05, 0.5])
    # set the color of each line according to its chlorophyll level
    for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
        line.set_color(color_map(map_norm(data["Avg Total Chlorophyll (µg/cm2)"]))[i])
    # make mean line
    ax.plot(x, x_data.mean(), color="black", label="mean")
    # return the ScalarMappable that can be used to make a color bar

    return mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)


def visualize_2_sensor_raw_data(save_filename: str = ""):
    """ Make a graph to visualize the AS7262 and AS7263 raw data.

    Make a 2x2 graph with each sensor as a column and the reflectance on the top row
    and the raw data counts on the bottom row.
    Shows the mean in black and all the other measurements in a shade of green that
    correlates to the level of chlorophyll of the leaf measured

    Args:
        save_filename (str): Name to save the file to, if falsy then no file is saved.

    Returns:
        None, displays the graph, or can be changed to save the image

    """
    figure, axs = plt.subplots(nrows=2, ncols=2,
                               sharex="col", figsize=(7.5, 4))
    for i, sensor in enumerate(["as7262", "as7263"]):
        print(f"i = {i}")
        visualize_raw_data(ax=axs[0, i], sensor=sensor)
        axs[0, i].set_title(f"{sensor.upper()}\nReflectance")
        axs[0, i].set_ylabel("% reflectance")
        color_map = visualize_raw_data(ax=axs[1, i],
                                       measurement_type="raw",
                                       sensor=sensor)
        axs[1, i].set_title("Raw Measurements")
        axs[1, i].set_xlabel("Wavelength (nm)")
        axs[1, i].set_ylabel("Counts (1000s)")
    # adjust the plots to look good
    plt.subplots_adjust(right=SUBPLOT_RIGHT_PAD,
                        wspace=WIDTH_PADDING)
    # add the color bar now
    color_bar_axis = figure.add_axes(COLOR_BAR_AXIS)
    figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                    label="Total Chlorophyll (µg/cm2)", fraction=0.08)
    if save_filename:
        figure.savefig(save_filename)


def visualize_3_sensor_raw_data(save_filename: str = ""):
    """ Make a graph to visualize the AS7262, AS7263 and AS7265x raw data.

    Make a 2x3 graph with each sensor as a column and the reflectance on the top row
    and the raw data counts on the bottom row. Shows the mean in black and all the other
    measurements in a shade of green that correlates to the level of
    chlorophyll of the leaf measured

    Args:
        save_filename (str): Name to save the file to, if falsy then no file is saved.

    Returns:
        None, displays the graph, or can be changed to save the image

    """
    figure, axs = plt.subplots(nrows=2, ncols=3,
                               sharex="col", figsize=(7.5, 4))
    # make each figure
    for i, sensor in enumerate(["as7262", "as7263", "as7265x"]):
        print(f"i = {i}")
        led = None
        if sensor == 'as7265x':
            led = "b'White'"
        visualize_raw_data(ax=axs[0, i], sensor=sensor, led=led)
        axs[0, i].set_title(f"{sensor.upper()}\nReflectance")
        axs[0, i].set_ylabel("% reflectance")
        color_map = visualize_raw_data(ax=axs[1, i],
                                       measurement_type="raw",
                                       sensor=sensor, led=led)
        axs[1, i].set_title("Raw Measurements")
        axs[1, i].set_xlabel("Wavelength (nm)")
        axs[1, i].set_ylabel("Counts (1000s)")

    # add the color bar now
    plt.subplots_adjust(right=SUBPLOT_RIGHT_PAD,
                        wspace=WIDTH_PADDING)
    # add the color bar now
    color_bar_axis = figure.add_axes(COLOR_BAR_AXIS)
    figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                    label="Total Chlorophyll (µg/cm2)", fraction=0.08)
    if save_filename:
        figure.savefig(save_filename)


def visualize_as7265x_different_leds(save_filename: str = ""):
    pass


if __name__ == '__main__':

    if True:
        visualize_2_sensor_raw_data(save_filename=
                                    "../../images/draft_spectrum/2_sensors_raw_data.svg")

        visualize_3_sensor_raw_data(save_filename=
                                    "../../images/draft_spectrum/3_sensors_raw_data.svg")
    else:
        visualize_2_sensor_raw_data()
        visualize_3_sensor_raw_data()
    plt.show()
