# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add functions to export the spectrum and chlorophyll data easily, this
should be the only file that needs to now the data files paths.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from functools import lru_cache
from pathlib import Path
import pickle

# installed libraries
import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).parent.parent.parent / "data"
RAW_SPECTRUM_FOLDER = DATA_FOLDER / "spectrum" / "raw"
REFLECTANCE_SPECTRUM_FOLDER = DATA_FOLDER / "spectrum" / "reflectance"
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
            mean: bool = False, read_numbers: int = None,
            send_leaf_numbers=False,
            ) -> (tuple[pd.DataFrame, pd.DataFrame] |
                  tuple[pd.DataFrame, pd.DataFrame, pd.Series]):
    """ Get the data for a sensor leaf combination.

    Get a data set based on the leaf, sensor, which type (raw data counts or reflectance)
    the integration time, led, and led current.

    Args:
        sensor (str): Which sensor to get the data for "as7262", "as7263", or "as7265x"
        leaf (str): Which leaf to get the data for, works for "mango", "banana",
        "jasmine", "rice", "sugarcane"
        measurement_type (str):  Which type of data to get "raw" for data counts,
        "reflectance" for reflectance values, or "absorbance".
        chloro_columns (str):  Which chlorophyll columns to get, will have chlorophyll
        by area or weight. Valid inputs are "all", "area", or "weight"
        int_time (int): Integration time for the sensor read
        led (str): LED used to measure
        led_current (str): Current of the LED used in the measurement
        mean (bool): If the leaf measurements should be averaged
        read_numbers (int): If int, will only return the read number given, if none,
        all reads are returned

    Returns:
        - pd.DataFrame: DataFrame of the spectra channels for the given conditions
        - pd.DataFrame: targets for fitting the spectrum, different chlorophyll measurements
        - pd.DataFrame: DataFrame of leaf numbers, used for grouping

    """
    print("get data args: ")
    print(sensor, leaf, measurement_type, int_time, led, led_current, read_numbers, mean)
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
                    measurement_mode=measurement_type)

    # error check the integration time, led, and led current are in the data set
    # and get only the data with those values
    if int_time not in data["integration time"].unique():
        raise ValueError(f"'{int_time}' is not a valid integration time, valid values are:"
                         f"{data['integration time'].unique()}")
    data = data[data["integration time"] == int_time]

    if sensor == "as7265x" and led == 'White LED':
        led = "b'White'"

    if led not in data["led"].unique():
        raise ValueError(f"'{led}' is not a valid led, valid values are:"
                         f"{data['led'].unique()}")
    data = data[data["led"] == led]

    if led_current not in data["led current"].unique():
        raise ValueError(f"'{led_current}' is not a valid led, valid values are:"
                         f"{data['led current'].unique()}")
    data = data[data["led current"] == led_current]
    if mean:
        data = data.groupby("Leaf No.").mean(numeric_only=True)

    if read_numbers:
        data = data[data["Read number"] == read_numbers]

    x_columns = []
    for column in data.columns:
        if 'nm' in column:
            x_columns.append(column)
    # set the index to the Leaf number
    groups = data["Leaf No."]
    x_data = data[x_columns]

    if measurement_type == "absorbance":
        print("taking absorbance")
        x_data = -np.log10(x_data)

    if send_leaf_numbers:
        return x_data, data[chloro_columns], groups
    else:
        return x_data, data[chloro_columns]


@lru_cache()  # cache the data reads, helps make tests much shorter
def get_data(sensor: str, leaf: str, measurement_mode: str,
             mean: bool = False) -> pd.DataFrame:
    """ Get the data for a sensor, leaf, measurement type combination.

    Args:
        sensor (str): Which sensor to get the data for "as7262", "as7263", or "as7265x"
        leaf (str): Which leaf to get the data for, works for "mango", "banana",
        "jasmine", "rice", "sugarcane"
        measurement_mode (str):  Which type of data to get "raw" for data counts or
        "reflectance" for reflectance values
        mean (bool): If the leaf measurements should be averaged

    Returns:
        pd.DataFrame: DataFrame of all data for the given sensor, leaf and measurement type.

    """
    if measurement_mode == "raw":
        data_path = RAW_SPECTRUM_FOLDER
    elif measurement_mode == "reflectance":
        data_path = REFLECTANCE_SPECTRUM_FOLDER
    elif measurement_mode == "absorbance":  # get reflectance and then calculate ab
        data_path = REFLECTANCE_SPECTRUM_FOLDER
    else:
        raise ValueError(f"type argument must be 'raw', 'absorbance' or 'reflectance, "
                         f"'{measurement_mode}' is not value")
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


def get_data_slices(df: pd.DataFrame, selected_column: str,
                    values: list):
    """ Take a DataFrame and return a sub-slice with the selected
    values in the selected columns.

    Just calls .isin method, but this is easier to remember and to test

    Args:
        df (pd.DataFrame): DataFrame to get slices from
        selected_column (str): column name to look for the values in
        values (list): values to select

    Returns:
        pd.DataFrame: a subset of the DataFrame passed in with the proper rows
        selected

    """
    new_df = df.loc[df[selected_column].isin(values)]
    return new_df


def get_cleaned_data(sensor: str, leaf: str, pickle_file: str = "final_dataset.pkl",
                     mean: bool = False):
    """
    Load cleaned data from a pickle file and retrieve the data for a specified sensor and leaf.
    All data is the absorbance.

    Parameters:
    ----------
    sensor : str
        The name of the sensor (e.g., 'as7265x').
    leaf : str
        The identifier of the leaf (e.g., 'mango').
    pickle_file : str, optional
        The path to the pickle file containing the cleaned dataset,
        by default 'final_dataset.pkl'.
    mean: bool, optional
        Boolean to specify if the spectrum data should be averaged.

    Returns:
    -------
    tuple
        A tuple (x, y, groups), where:
        - x (pd.DataFrame): Feature data for the specified sensor and leaf.
        - y (pd.Series): Target data for the specified sensor and leaf.
        - groups (pd.Series): Group information for the specified sensor and leaf.

    Raises:
    -------
    FileNotFoundError:
        If the specified pickle file does not exist.
    KeyError:
        If the specified sensor or leaf is not in the dataset.
    """
    try:
        # Load the pickled dataset
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Retrieve the data for the specified sensor and leaf
        x_columns = None # intialize incase they are gettting single AS7265x dataset
        if sensor in ["as72651", "as72652", "as72653"]:  # if single AS7265x dataset do:
            sensor_data = data.get("as7265x")  # have to get full data first
            # then select wavelenghts to get later
            if sensor == "as72651":
                x_columns = ["610 nm", "680 nm", "730 nm", "760 nm", "810 nm", "860 nm"]
            elif sensor == "as72652":
                x_columns = ["560 nm", "585 nm", "645 nm", "705 nm", "900 nm", "940 nm"]
            elif sensor == "as72653":
                x_columns = ["410 nm", "435 nm", "460 nm", "485 nm", "510 nm", "535 nm"]
        else:
            sensor_data = data.get(sensor)

        if sensor_data is None:
            raise KeyError(f"Sensor '{sensor}' not found in the dataset.")

        leaf_data = sensor_data.get(leaf)
        if leaf_data is None:
            raise KeyError(f"Leaf '{leaf}' not found under sensor '{sensor}' in the dataset.")

        if mean:
            x = leaf_data["x"].set_index(leaf_data["groups"])
            x = x.groupby(["Leaf No."]).mean(numeric_only=True)
            y = leaf_data["y"].set_index(leaf_data["groups"])
            y = y.groupby(["Leaf No."]).mean(numeric_only=True)
            return x, y, x.index
        # Extract x, y, and groups
        x = leaf_data["x"]
        if x_columns:
            x = x[x_columns]
        # print(x.head())
        y = leaf_data["y"]
        # print(y.head())
        groups = leaf_data["groups"]

        return x, y, groups

    except FileNotFoundError:
        raise FileNotFoundError(f"The specified pickle file '{pickle_file}' does not exist.")
    except KeyError as e:
        raise e



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
