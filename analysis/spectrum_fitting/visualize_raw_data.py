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
plt.style.use('seaborn-v0_8')
COLOR_MAP = 'Greens'  # options: Greens, YlGn
COLOR_MAP_FROM_LIST = ["palegreen", "darkgreen"]
INT_TIME = 100  # the integration time to display
CURRENT = "25 mA"  # the current to use in the measurement to display
COLOR_BAR_AXIS = [.90, .1, 0.02, 0.8]
AS7265X_COLOR_BAR_AXIS = [.90, .25, 0.02, 0.5]
SUBPLOT_RIGHT_PAD = 0.87
WIDTH_PADDING = 0.3
REFLECTANCE_YLABEL = "% reflectance"
RAW_DATA_YLABEL = "Counts (1000s)"
# REFERENCES
# AS7265x unique() leds: ["b'IR'", "b'UV IR'", "b'UV'", "b'White IR'",
# "b'White UV IR'", "b'White UV'", "b'White'"]


# helper functions
@lru_cache()  # this gets called 5-7 times so just memorize it instead of making a class
def make_color_map(color_min: float, color_max: float
                   ) -> tuple[mpl.colors.LinearSegmentedColormap,
                              mpl.colors.Normalize]:
    """ Make a linear segment color map from a list

    Args:
        color_min (float): Minimum value for colormap normalization.
        color_max (float): Maximum value for colormap normalization.

    Returns:
        tuple: A tuple containing:
            - mpl.colors.LinearSegmentedColormap: Linear segmented colormap object.
            - mpl.colors.Normalize: Normalization object for the colormap.

        Examples:
        >>> data = get_data.get_data()
        >>> lines = plt.plot(x, y)
        >>> color_map, map_norm = make_color_map(data["Avg Total Chlorophyll (µg/cm2)"].min(),
        ...                                      data["Avg Total Chlorophyll (µg/cm2)"].max())
        >>> for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
        ...     line.set_color(color_map(map_norm(data["Avg Total Chlorophyll (µg/cm2)"]))[i])

    """
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
                             measurement_mode=measurement_type,
                             mean=True)
    print(data["led"].unique())
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
        None, displays the graph, or saves the image

    """
    figure, axs = plt.subplots(nrows=2, ncols=2,
                               sharex="col", figsize=(7.5, 4))
    for i, sensor in enumerate(["as7262", "as7263"]):
        print(f"i = {i}")
        visualize_raw_data(ax=axs[0, i], sensor=sensor)
        axs[0, i].set_title(f"{sensor.upper()}\nReflectance")
        axs[0, i].set_ylabel(REFLECTANCE_YLABEL)
        color_map = visualize_raw_data(ax=axs[1, i],
                                       measurement_type="raw",
                                       sensor=sensor)
        axs[1, i].set_title("Raw Measurements")
        axs[1, i].set_xlabel("Wavelength (nm)")
        axs[1, i].set_ylabel(RAW_DATA_YLABEL)
    # adjust the plots to look good
    plt.subplots_adjust(right=SUBPLOT_RIGHT_PAD,
                        wspace=WIDTH_PADDING)
    # add the color bar now
    color_bar_axis = figure.add_axes(COLOR_BAR_AXIS)
    figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                    label="Total Chlorophyll (µg/cm\u00b2)", fraction=0.08)
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
        None, displays the graph, or saves the image

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
        axs[0, i].set_ylabel(REFLECTANCE_YLABEL)
        color_map = visualize_raw_data(ax=axs[1, i],
                                       measurement_type="raw",
                                       sensor=sensor, led=led)
        axs[1, i].set_title("Raw Measurements")
        axs[1, i].set_xlabel("Wavelength (nm)")
        axs[1, i].set_ylabel(RAW_DATA_YLABEL)

    # add the color bar now
    plt.subplots_adjust(right=SUBPLOT_RIGHT_PAD,
                        wspace=WIDTH_PADDING)
    # add the color bar now
    color_bar_axis = figure.add_axes(COLOR_BAR_AXIS)
    figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                    label="Total Chlorophyll (µg/cm\u00b2)", fraction=0.08)
    if save_filename:
        figure.savefig(save_filename)


def visualize_as7265x_different_leds(leds: list[str], save_filename: str = ""):
    """ Make a graph to visualize the raw data for a list of LEDS for the AS7265x color sensor

    Args:
        leds (list[str]): list strings of the LEDS to display, this has to the string used in
        the data files in the "led" column.
        save_filename (str): Name to save the file to, if falsy then no file is saved.

    Returns:
        None, displays the graph, or saves the image

    """
    figure, axs = plt.subplots(nrows=len(leds), ncols=2,
                               sharex="col", figsize=(7.5, 2.75*len(leds)))
    for i, led in enumerate(leds):
        print(f"i = {i}")
        visualize_raw_data(ax=axs[i, 0], sensor="as7265x", led=led,
                           measurement_type="reflectance")
        axs[i, 0].set_ylabel(REFLECTANCE_YLABEL)
        # because I didn't fix that led being saved as a byte string earlier
        # now you have to split the LED string out
        # axs[i, 0].set_title("  " + led.split("'")[1]+" LED", loc="left", y=1, pad=-14)
        axs[i, 0].annotate(led.split("'")[1]+" LED", (.08, .85), xycoords='axes fraction')
        color_map = visualize_raw_data(ax=axs[i, 1], sensor="as7265x", led=led,
                                       measurement_type="raw")
        axs[i, 1].set_ylabel(RAW_DATA_YLABEL)
    # make titles at top of columns
    axs[0, 0].set_title("Reflectance")
    axs[0, 1].set_title("Raw data counts")

    # add the color bar now
    plt.subplots_adjust(right=SUBPLOT_RIGHT_PAD)
    color_bar_axis = figure.add_axes(AS7265X_COLOR_BAR_AXIS)
    figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                    label="Total Chlorophyll (µg/cm\u00b2)", fraction=0.08)
    if save_filename:
        figure.savefig(save_filename)


if __name__ == '__main__':
    # if True:
    #     visualize_2_sensor_raw_data(save_filename=
    #                                 "../../images/draft_spectrum/2_sensors_raw_data.svg")
    #
    #     visualize_3_sensor_raw_data(save_filename=
    #                                 "../../images/draft_spectrum/3_sensors_raw_data.svg")
    # else:
    #     visualize_2_sensor_raw_data()
    #     visualize_3_sensor_raw_data()
    # if True:
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'White'", "b'IR'"], save_filename=
    #     "../../images/draft_spectrum/as7265x_raw_data_white_ir_uv.svg")
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'IR'"], save_filename=
    #     "../../images/draft_spectrum/as7265x_raw_data_ir_uv.svg")
    # else:
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'White'", "b'IR'"])
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'IR'"])
    plt.show()
