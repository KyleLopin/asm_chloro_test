# Copyright (c) 2023-5 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make functions to add the average Total chlorophyll, chlorophyll a,
and chlorophyll b levels to the spectrum files.

Make the join call into a function to test.
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path

# installed libraries
import pandas as pd

# local files

# make file paths
CHLORO_DATA_FOLDER = (Path(__file__).parent.parent / "data" / "chlorophyll_data"
                      / "summary_data")
SPECTRUM_DATA_FOLDER = (Path(__file__).parent.parent / "data" / "spectrum_data"
                        / "raw_data")
REFLECTANCE_DATA_FOLDER = (Path(__file__).parent.parent / "data" / "spectrum_data"
                           / "reflectance_data")

ALL_LEAVES = tuple(("mango", "banana", "jasmine", "rice", "sugarcane"))
ALL_SENSORS = tuple(("as7265x", "as7262", "as7263"))
# show all columns to check the data
pd.set_option('display.max_columns', None)
pd.options.display.width = None


def add_chloro_to_df(chloro_df: pd.DataFrame, spectrum_df: pd.DataFrame) -> pd.DataFrame:
    """ Join the chlorophyll data with the spectrum data, add all columns based on index.

    Ended up just being a join call, but can use for testing.

    Args:
        chloro_df (pd.DataFrame): first DataFrame to join together
        spectrum_df(pd.DataFrame): second DataFrame to join together

    Returns:
        pd.DataFrame: joined DataFrames
    """
    return spectrum_df.join(chloro_df)


if __name__ == '__main__':
    # go through each leaf
    for leaf in ALL_LEAVES:
        # get chlorophyll data
        chloro_filename = CHLORO_DATA_FOLDER / f"{leaf}_summary.csv"
        chloro_data = pd.read_csv(chloro_filename)
        # some files have a missed labeled leaf number column, this fixes it
        if "Leaf number" in chloro_data.columns:
            chloro_data = chloro_data.rename(columns={"Leaf number": "Leaf No."})

        # both DataFrames need to have their indexes set to "Leaf No." to join them
        chloro_data.set_index("Leaf No.", inplace=True)
        # go through each sensor to update the fields
        for sensor in ALL_SENSORS:
            # get spectrum filename and data
            spectrum_filename = SPECTRUM_DATA_FOLDER / f"{leaf}_{sensor}_data.csv"
            spectrum_data = pd.read_csv(spectrum_filename)
            # some files have a missed labeled leaf number column, this fixes it
            if "Leaf number" in spectrum_data.columns:
                spectrum_data = spectrum_data.rename(columns={"Leaf number": "Leaf No."})
            # set index to join correctly
            spectrum_data.set_index("Leaf No.", inplace=True)
            # combine the data
            # combined_data = add_chloro_to_df(chloro_data, spectrum_data)

            # overwrite file with new data, the originals are saved in a zip
            # combined_data.to_csv(SPECTRUM_DATA_FOLDER / f"{leaf}_{sensor}_data.csv")

            # repeat the process for the reflectance data
            reflectance_filename = REFLECTANCE_DATA_FOLDER / f"{leaf}_{sensor}_data.csv"
            reflectance_data = pd.read_csv(reflectance_filename)
            if "Leaf number" in reflectance_data.columns:
                reflectance_data = reflectance_data.rename(columns={"Leaf number": "Leaf No."})

            reflectance_data.set_index("Leaf No.", inplace=True)
            # print(reflectance_data)
            # print(chloro_data)
            refl_combined_data = add_chloro_to_df(chloro_data, reflectance_data)
            print(refl_combined_data)
            refl_combined_data.to_csv(REFLECTANCE_DATA_FOLDER / f"{leaf}_{sensor}_data.csv")
