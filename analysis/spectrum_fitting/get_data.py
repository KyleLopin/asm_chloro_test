# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add functions to export the spectrum and chlorophyll data easily, this
should be the only file that needs to now the data files paths.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache
from pathlib import Path

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"
RAW_SPECTRUM_FOLDER = DATA_FOLDER / "spectrum_data" / "raw_data"
REFLECTANCE_SPECTRUM_FOLDER = DATA_FOLDER / "spectrum_data" / "reflectance_data"
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
ALL_SENSORS = ["as7265x", "as7262", "as7263"]
ALL_CHLORO = ['Avg Total Chlorophyll (µg/cm2)', 'Avg Chlorophyll a (µg/cm2)',
              'Avg Chlorophyll b (µg/cm2)', 'Avg Total Chlorophyll (µg/mg)',
              'Avg Chlorophyll a (µg/mg)', 'Avg Chlorophyll b (µg/mg)']
AREA_CHLORO = tuple(('Avg Total Chlorophyll (µg/cm2)', 'Avg Chlorophyll a (µg/cm2)',
                     'Avg Chlorophyll b (µg/cm2)'))
WEIGHT_CHLORO = tuple(('Avg Total Chlorophyll (µg/mg)', 'Avg Chlorophyll a (µg/mg)',
                       'Avg Chlorophyll b (µg/mg)'))
pd.set_option('display.max_columns', None)
pd.options.display.width = None


def get_x_y(sensor: str, leaf: str, measurement_type: str,
            chloro_columns: str = "all", int_time: int = 150,
            led: str = "White LED", led_current: str = "12.5 mA",
            mean: bool = False
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Get the data for a sensor leaf combination.

    Get a data set based on the leaf, sensor, which type (raw data counts or reflectance)
    the integration time, led, and led current.

    Args:
        sensor (str): Which sensor to get the data for "as7262", "as7263", or "as7265x"
        leaf (str): Which leaf to get the data for, works for "mango", "banana",
        "jasmine", "rice", "sugarcane"
        measurement_type (str):  Which type of data to get "raw" for data counts or
        "reflectance" for reflectance values
        chloro_columns (str):  Which chlorophyll columns to get, will have chlorophyll
        by area or weight. Valid inputs are "all", "area", or "weight"
        int_time (int): Integration time for the sensor read
        led (str): LED used to measure
        led_current (str): Current of the LED used in the measurement
        mean (bool): If the leaf measurements should be averaged

    Returns:
        - pd.DataFrame: DataFrame of the spectra channels for the given conditions
        - pd.DataFrame: targets for fitting the spectrum, different chlorophyll measurements

    """
    if leaf not in ALL_LEAVES:
        raise ValueError(f"leaf '{leaf}' is not valid, must be one of these: {ALL_LEAVES}")
    if sensor not in ALL_SENSORS:
        raise ValueError(f"sensor '{sensor}' is not valid, must be one of these: {ALL_SENSORS}")
    if chloro_columns == "all":
        chloro_columns = ALL_CHLORO
    elif chloro_columns == "area":
        chloro_columns = AREA_CHLORO
    elif chloro_columns == "weight":
        chloro_columns = WEIGHT_CHLORO
    else:
        raise ValueError(f"chloro_columns '{chloro_columns}' is not valid, must be:"
                         f"'all', 'area', or 'weight'")

    data = get_data(sensor=sensor, leaf=leaf,
                    measurement_type=measurement_type)

    # error check the integration time, led, and led current are in the data set
    # and get only the data with those values
    if int_time not in data["integration time"].unique():
        raise ValueError(f"'{int_time}' is not a valid integration time, valid values are:"
                         f"{data['integration time'].unique()}")
    data = data[data["integration time"] == int_time]

    if led not in data["led"].unique():
        raise ValueError(f"'{led}' is not a valid led, valid values are:"
                         f"{data['led'].unique()}")
    data = data[data["led"] == led]

    if led_current not in data["led current"].unique():
        raise ValueError(f"'{led_current}' is not a valid led, valid values are:"
                         f"{data['led current'].unique()}")
    data = data[data["led current"] == led_current]
    if mean:
        data = data.groupby("Leaf No.")

    x_columns = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)
    # print(f"x columns: {x_columns}")
    return data[x_columns], data[chloro_columns]


@lru_cache()  # cache the data reads, helps make tests much shorter
def get_data(sensor: str, leaf: str, measurement_type: str,
             mean: bool = False) -> pd.DataFrame:
    """ Get the data for a sensor, leaf, measurement type combination.

    Args:
        sensor (str): Which sensor to get the data for "as7262", "as7263", or "as7265x"
        leaf (str): Which leaf to get the data for, works for "mango", "banana",
        "jasmine", "rice", "sugarcane"
        measurement_type (str):  Which type of data to get "raw" for data counts or
        "reflectance" for reflectance values
        mean (bool): If the leaf measurements should be averaged

    Returns:
        pd.DataFrame: DataFrame of all data for the given sensor, leaf and measurement type.

    """
    if measurement_type == "raw":
        data_path = RAW_SPECTRUM_FOLDER
    elif measurement_type == "reflectance":
        data_path = REFLECTANCE_SPECTRUM_FOLDER
    else:
        raise ValueError(f"type argument must be 'raw' or 'reflectance, "
                         f"'{measurement_type}' is not value")
    filename = data_path / f"{leaf}_{sensor}_data.csv"
    data = pd.read_csv(filename)
    if mean:
        data = data.groupby(["Leaf No.", "integration time",
                             "led", "led current"], as_index=False
                            ).mean(numeric_only=True)
    return data


def get_options(sensor: str, leaf: str, measurement_type: str
                ) -> tuple[list[int], list[str], list[str]]:
    """ Get the integration time, LEDs and LED currents used an experiment.

    Args:
        sensor (str): Which sensor to get the data for "as7262", "as7263", or "as7265x"
        leaf (str): Which leaf to get the data for, works for "mango", "banana",
        "jasmine", "rice", "sugarcane"
        measurement_type (str):  Which type of data to get "raw" for data counts or
        "reflectance" for reflectance values

    Returns:
        - list[int]: list of the integer values for the integration times used in the experiments
        - list[str]: list of LEDs used in the experiments
        - list[str]: list of currents used for the LEDs used in the experiments
    """
    data = get_data(sensor, leaf, measurement_type)
    return (list(data["integration time"].unique()),
            list(data["led"].unique()),
            list(data["led current"].unique()))


if __name__ == '__main__':
    # Test the function works
    # x, y = get_x_y(sensor="as7262", leaf="banana",
    #                 type="raw", chloro_columns="all",
    #                 int_time=150, led="White LED")
    # print(x)
    # print('=====')
    # print(y)
    # plt.plot(x.T)
    # plt.show()
    int_times, leds, currents = get_options(sensor="as7262", leaf="banana",
                                            measurement_type="raw")
    print(int_times)
    print(leds)
    print(currents)