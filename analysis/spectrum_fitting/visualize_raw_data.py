# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
To make Figure 5 of manuscript use this function: visualize_raw_and_reflectance

To make Figure 7 use: visualize_4_leaves_3_sensors
Make functions to visualize data from as7262, as7263 and as7265x color sensor data
for the chlorophyll data.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache

# installed libraries
import matplotlib as mpl
from matplotlib.lines import Line2D  # for typehinting
import matplotlib.pyplot as plt

# local files
import remove_outliers
import get_data

# plt.style.use('seaborn-v0_8')
COLOR_MAP = 'Greens'  # options: Greens, YlGn
COLOR_MAP_FROM_LIST = ["palegreen", "darkgreen"]
INT_TIME = 50  # the integration time to display
CURRENT = "12.5 mA"  # the current to use in the measurement to display
COLOR_BAR_AXIS = [.90, .1, 0.02, 0.8]
AS7265X_COLOR_BAR_AXIS = [.90, .25, 0.02, 0.5]
SENSORS = ["as7262", "as7263", "as7265x"]
SUBPLOT_RIGHT_PAD = 0.87
WIDTH_PADDING = 0.3
REFLECTANCE_YLABEL = "% reflectance"
RAW_DATA_YLABEL = "Counts (1000s)"
FRUIT = "mango"
OTHER_LEAVES = ["banana", "jasmine", "mango", "rice", "sugarcane"]
OTHER_LEAVES.remove(FRUIT)
print(OTHER_LEAVES)
plt.rcParams['font.family'] = 'Arial'
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
            x, y = get_data.get_x_y(sensor="as7262", leaf="mango", measurement_type="reflectance")
            lines = plt.plot(x, y)
            color_map, map_norm = make_color_map(y["Avg Total Chlorophyll (µg/cm2)"].min(),
                                                 y["Avg Total Chlorophyll (µg/cm2)"].max())
            for i, line in enumerate(lines):  # type: int, mpl.lines.Line2D
                line.set_color(tuple(color_map(map_norm(y["Avg Total Chlorophyll (µg/cm2)"]))[i]))

    """
    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "", COLOR_MAP_FROM_LIST)

    map_norm = mpl.colors.Normalize(vmin=color_min, vmax=color_max)
    return color_map, map_norm


def visualize_raw_data(ax: plt.Axes = None, sensor: str = "as7262",
                       leaf: str = FRUIT, led: str = "",
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
    print(x_columns)
    x_data = data[x_columns]
    if measurement_type == 'raw':  # make the y labels smaller
        x_data = x_data / 1000
    # make color map
    color_map, map_norm = make_color_map(data["Avg Total Chlorophyll (µg/cm2)"].min(),
                                         data["Avg Total Chlorophyll (µg/cm2)"].max())

    lines = ax.plot(x, x_data.T, alpha=0.7)  # type: list[Line2D]
    # set the color of each line according to its chlorophyll level
    for i, line in enumerate(lines):  # type: int, Line2D
        line.set_color(color_map(map_norm(data["Avg Total Chlorophyll (µg/cm2)"]))[i])
    # make mean line
    ax.plot(x, x_data.mean(), color="black", label="mean")
    # return the ScalarMappable that can be used to make a color bar

    return mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)


def visualize_4_leaves_3_sensors():
    """ Function to make manuscript figure 6

    Visualize reflectance data for 4 different leaves and 3 sensors, generating
    12 subplots in a 6x2 grid. Each plot represents the reflectance spectrum
    for a specific leaf and sensor combination, with lines colored based on
    chlorophyll levels.

    Shared x-axes for wavelength and y-axes are scaled based on the sensor to better compare
    the spectrum between sensors.
    The color map for the  reflectance lines is based on the chlorophyll levels,
    with mean lines displayed in black for each plot.
    A color bar indicates the chlorophyll levels.

    Returns:
    --------
    None
        Displays the reflectance plots and adds a color bar, or saves the figure.
        Use line comments to change between displaying or saving the figure.

    Notes:
    ------
    - Reflectance is normalized and plotted for wavelengths 410 to 940 nm.
    - Labels on the x-axis are selectively displayed to avoid clutter.
    - The function dynamically adjusts subplot settings to optimize space.
    """
    plt.style.use('seaborn-v0_8')
    fig, axs = plt.subplots(6, 2, sharex="col",
                            figsize=(7.5, 8))
    # make a color map to color code the chlorophyll levels plotted
    color_map, map_norm = make_color_map(0, 100)
    j = 0
    # 4 leaves and 3 sensors so 12 graphs
    for i in range(12):
        leaf = OTHER_LEAVES[i//3]  # rotate through the leaves
        sensor = SENSORS[i % 3]  # rotate through the sensors
        if i == 6:
            j = 1  # first 5 are left column, at 6 increment to seconds column
        led = "White LED"

        # axs[i % 6][j].annotate(f"AS{sensor[2:]}", (0.02, 0.90), xycoords='axes fraction',
        #                        fontsize=12, fontweight='bold', va='top')
        axs[i % 6][j].annotate(f"{chr(i+97)})", (0.02, 0.92), xycoords='axes fraction',
                               fontsize=14, fontweight='bold', va='top')
        sensor_coords = (0.65, 0.20)
        int_time = 50
        if sensor == "as7262":
            axs[i % 6][j].annotate(leaf.capitalize(), (0.65, 0.80), xycoords='axes fraction',
                                   fontsize=12, fontweight='bold', va='top')
            axs[i % 6][j].set_ylim([0, 0.45])
        elif sensor == "as7265x":
            led = "b'White IR'"
            axs[i % 6][j].set_ylim([0, 0.68])
        else:  # AS7263
            axs[i % 6][j].set_ylim([0, 0.90])
            int_time = 250
            # sensor_coords = (0.10, 0.20)
        # add caption for each sensor
        # axs[i % 6][j].annotate(f"AS{sensor[2:]}", sensor_coords, xycoords='axes fraction',
        #                        fontsize=10, fontweight='bold', va='top')

        x, y, groups = get_data.get_x_y(sensor=sensor, leaf=leaf, measurement_type="reflectance",
                                        int_time=int_time, led=led, led_current="12.5 mA",
                                        send_leaf_numbers=True)
        y = y["Avg Total Chlorophyll (µg/cm2)"]
        wavelengths = x.columns
        x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]
        # we want to color the lines on chlorophyll levels
        lines = axs[i % 6][j].plot(x_wavelengths, x.T, alpha=0.7, lw=1)  # type: list[mpl.lines.Line2D]
        # TODO: How to find outliers and add here?
        data_mask = remove_outliers.remove_outliers_from_residues(x, y, groups)
        # print(data_mask)
        # set the color of each line according to its chlorophyll level
        for k, line in enumerate(lines):  # type: int, mpl.lines.Line2D
            if data_mask[k]:  # not outlier color code
                line.set_color(color_map(map_norm(y))[k])
                line.set_zorder(1)
            else:
                line.set_color("red")
                line.set_zorder(2)
        # make mean line
        axs[i % 6][j].plot(x_wavelengths, x.mean(), color="black", label="Mean")
        if i == 4:
            # axs[i][0].plot([], [], color=color_map(map_norm(50)), label="Leaf Spectrum")
            axs[i][0].plot([], [], color="red", label="Outlier")
            axs[i][0].legend(loc='lower left', frameon=False)

    # 18 wavelengths is too much to show on the axis so skip some labels
    skip_index = {1, 3, 5, 7, 10, 12}
    # wavelenghts and x_wavelengths are set to AS7265x sensor as that was the last
    # sensor in the for loop
    labels = [wavelength if i not in skip_index else ''
              for i, wavelength in enumerate(wavelengths)]
    for z in [0, 1]:
        # set the wavelength names at the ticks
        axs[5][z].set_xticks(ticks=x_wavelengths, labels=labels, rotation=60)
        # tighten up the axis to remove the extra room in default adds
        axs[5][z].set_xlim([410, 940])
    fig.suptitle("Leaf reflectance")

    plt.tight_layout()
    fig.text(0.04, 0.5, 'Normalized Reflectance',
             ha='center', va='center', rotation='vertical', fontsize=12)

    # Use subplots_adjust to add more space on the left side for the y-axis label
    fig.subplots_adjust(left=0.1, right=0.87)

    # convert color_map from LinearSegmentedColormap to ScalarMappable
    color_map = mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)
    color_bar_axis = fig.add_axes(COLOR_BAR_AXIS)
    color_bar = fig.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                             fraction=0.08)
    # Adjust the label padding (distance from the color bar)
    color_bar.set_label(r'Total Chlorophyll ($\mu$g/cm$^2$)',
                        labelpad=-1)
    fig.savefig("reflectance_data_w_outliers.jpeg", dpi=600)
    plt.show()


def visualize_raw_and_reflectance():
    """ Function to make figure for manuscript with the 4 sensor / LED conditions of
    AS7262, AS7263, AS7265x + White LED, and AS7265x + White + IR LED on each row.
    Display the raw data on the left column, and on the reflectance on the right column.
    Align the wavelengths for the 3 sensors for visual effect

    Returns:
        None, show of save the figure in the function

    """
    leaf = FRUIT
    figure, axs = plt.subplots(nrows=4, ncols=2,
                               sharex="col", figsize=(7, 7))
    sensors = ["as7262", "as7263", "as7265x", "as7265x"]
    leds = ["White LED", "White LED", "b'White'", "b'White IR'"]
    color_map, map_norm = make_color_map(0, 100)

    for i, (sensor, led) in enumerate(zip(sensors, leds)):
        # add a-d on left figures
        axs[i][0].annotate(f"{chr(i+97)})", (0.02, 0.95), xycoords='axes fraction',
                           fontsize=14, fontweight='bold', va='top')
        # add e-h on right side figures
        axs[i][1].annotate(f"{chr(i + 101)})", (0.02, 0.95), xycoords='axes fraction',
                           fontsize=12, fontweight='bold', va='top')
        # add sensor names to figures
        axs[i][0].annotate(f"AS{sensor[2:]}", (0.50, 0.90), xycoords='axes fraction',
                           fontsize=12, fontweight='bold', va='top')
        led_str = str(led)
        if sensor == "as7265x":
            led_str = led_str[2:-1].replace(" ", " + ") + " LED"
        # add LED to figures
        axs[i][0].annotate(f"{led_str}", (0.50, 0.75), xycoords='axes fraction',
                           fontsize=12, fontweight='bold', va='top')
        for j, mode in enumerate(["raw", "reflectance"]):
            x, y = get_data.get_x_y(sensor=sensor, leaf=leaf, measurement_type=mode, int_time=50,
                                    led=led, led_current="12.5 mA")
            if sensor == "as7262":
                conversion = 45
            else:
                conversion = 35  # AS7265x and AS7263 both have lower conversion number
            if j == 0:  # The raw reflectance saved counts, convert to uW per cm2 per second
                x = x / conversion
            y = y["Avg Total Chlorophyll (µg/cm2)"]
            # to plot the wavelengths to scale convert to integers
            wavelengths = x.columns
            x_wavelengths = [int(wavelength.split()[0]) for wavelength in wavelengths]

            # we want to color the lines on chlorophyll levels
            lines = axs[i][j].plot(x_wavelengths, x.T, alpha=0.7)
            # set the color of each line according to its chlorophyll level
            for k, line in enumerate(lines):  # type: int, mpl.lines.Line2D
                line.set_color(color_map(map_norm(y))[k])

            # make mean line
            axs[i][j].plot(x_wavelengths, x.mean(), color="black", label="mean")

    # 18 wavelengths is too much to show on the axis so skip some labels
    skip_index = {1, 3, 5, 7, 10, 12}
    labels = [wavelength if i not in skip_index else ''
              for i, wavelength in enumerate(wavelengths)]

    for z in [0, 1]:
        # set the wavelength names at the ticks
        axs[3][z].set_xticks(ticks=x_wavelengths, labels=labels, rotation=60)
        # tighten up the axis to remove the extra room in default adds
        axs[3][z].set_xlim([410, 940])

    # make axis labels
    # axs[2][0].set_ylabel(r"Reflected light intensity ($\frac{\mu W}{cm^2 \cdot s}$)", fontsize=12)
    figure.text(0.04, 0.5, r"Light intensity ($\frac{\mu W}{cm^2 \cdot s}$)",
                ha='center', va='center', rotation='vertical', fontsize=12)
    # Add a y-axis label in the middle of the figure
    figure.text(0.47, 0.5, 'Normalized Reflectance',
                ha='center', va='center', rotation='vertical', fontsize=12)

    # add column titles
    axs[0][0].set_title("Raw reflected light intensity")
    axs[0][1].set_title('Relative Reflectance')
    figure.suptitle(f"{FRUIT.capitalize()} leaf reflectance", fontsize=14)

    # Apply tight_layout to automatically adjust subplot spacing
    plt.tight_layout()

    # Use subplots_adjust to add more space on the left side for the y-axis label
    figure.subplots_adjust(left=0.1, wspace=0.22, right=0.87)  # Increase the left margin

    # convert color_map from LinearSegmentedColormap to ScalarMappable
    color_map = mpl.cm.ScalarMappable(norm=map_norm, cmap=color_map)
    color_bar_axis = figure.add_axes(COLOR_BAR_AXIS)
    color_bar = figure.colorbar(color_map, cax=color_bar_axis, orientation="vertical",
                                fraction=0.08)
    # Adjust the label padding (distance from the color bar)
    color_bar.set_label(r'Total Chlorophyll ($\mu$g/cm$^2$)',
                        labelpad=-1)
    # plt.show()
    figure.savefig("raw_and_reflectance_data.jpeg", dpi=600)


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
        axs[i, 1].set_ylim([0, .1])
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
    if True:
        visualize_raw_and_reflectance()
        visualize_4_leaves_3_sensors()

    if False:
        visualize_3_sensor_raw_data()
    if False:
        visualize_as7265x_different_leds(leds=["b'UV'", "b'White'"])
        plt.show()
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'IR'"], save_filename=
    #     "../../images/draft_spectrum/as7265x_raw_data_ir_uv.svg")
    # else:
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'White'", "b'IR'"])
    #     visualize_as7265x_different_leds(leds=["b'UV'", "b'IR'"])
    # plt.show()
