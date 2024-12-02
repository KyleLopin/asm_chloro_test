# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Add unittest for functions in the analysis/spectrum_fitting/get_data.py file
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from pathlib import Path
import pickle
import unittest


# local files
import context
from analysis.spectrum_fitting.get_data import get_cleaned_data



data = {
    "banana": {"as7262": 296, "as7263": 298, "as7265x": 298},
    "jasmine": {"as7262": 300, "as7263": 298, "as7265x": 300},
    "mango": {"as7262": 300, "as7263": 297, "as7265x": 300},
    "rice": {"as7262": 299, "as7263": 296, "as7265x": 298},
    "sugarcane": {"as7262": 298, "as7263": 296, "as7265x": 296},
}


class TestGetCleanedData(unittest.TestCase):
    """
    Unit tests for the get_cleaned_data function.
    """

    def setUp(self):
        """
        Set up the testing environment by locating the pickle file.
        """
        # Locate the path to 'final_dataset.pkl' dynamically
        current_file = Path(__file__).resolve()  # Path to this test file
        project_root = current_file.parents[2]  # Adjust to reach project root
        self.pickle_file = project_root / "spectrum_fitting" / "final_dataset.pkl"

        if not self.pickle_file.exists():
            raise FileNotFoundError(f"Required pickle file '{self.pickle_file}' is missing.")


    def tearDown(self):
        """
        Clean up any temporary files after the tests are complete.
        """
        # No changes to the original dataset, so no cleanup is needed.

    def test_get_cleaned_data_all_combinations(self):
        """
        Test get_cleaned_data function for all sensor-leaf combinations.

        This function:
        - Loops through each sensor and leaf combination in the dataset.
        - Verifies that the shapes of x, y, and groups match expectations.
        - Ensures that the number of features in x aligns with the sensor type.
        """
        # Expected feature sizes for each sensor type
        feature_sizes = {
            "as7262": 6,
            "as7263": 6,
            "as7265x": 18,
        }

        # Load the dataset
        with open(self.pickle_file, "rb") as f:
            data = pickle.load(f)

        # Loop through each sensor and leaf in the dataset
        for sensor, leaves in data.items():
            for leaf, leaf_data in leaves.items():
                with self.subTest(sensor=sensor, leaf=leaf):
                    x, y, groups = get_cleaned_data(sensor, leaf, self.pickle_file)

                    # Validate the shape of x
                    self.assertEqual(x.shape[0], len(y), f"Row mismatch for {sensor}-{leaf}")
                    self.assertEqual(x.shape[1], feature_sizes[sensor], f"Feature mismatch for {sensor}-{leaf}")

                    # Validate the length of y and groups
                    self.assertEqual(len(y), len(leaf_data["y"]), f"y mismatch for {sensor}-{leaf}")
                    self.assertEqual(len(groups), len(leaf_data["groups"]), f"groups mismatch for {sensor}-{leaf}")

    def test_sensor_not_found(self):
        """
        Test get_cleaned_data raises a KeyError when an unknown sensor is provided.
        """
        with self.assertRaises(KeyError):
            get_cleaned_data("unknown_sensor", "banana", self.pickle_file)

    def test_leaf_not_found(self):
        """
        Test get_cleaned_data raises a KeyError when an unknown leaf is provided.
        """
        with self.assertRaises(KeyError):
            get_cleaned_data("as7262", "unknown_leaf", self.pickle_file)

    def test_pickle_file_not_found(self):
        """
        Test get_cleaned_data raises a FileNotFoundError when the pickle file is missing.
        """
        with self.assertRaises(FileNotFoundError):
            get_cleaned_data("as7262", "banana", "nonexistent.pkl")

