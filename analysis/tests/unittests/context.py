# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
add the proper directory to python path for tests to work
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import pathlib
import sys

chloro_file_folder = pathlib.Path().absolute().parent.parent / "chlorophyll_measurements"
spectrum_file_folder = pathlib.Path().absolute().parent.parent / "spectrum_fitting"
# Get the root directory of the project (assumes your tests folder is at the root level)
# project_root = pathlib.Path(__file__).resolve().parent.parent
# # Define the paths to the desired folders
# chloro_file_folder = project_root / "chlorophyll_measurements"
# spectrum_file_folder = project_root / "spectrum_fitting"

sys.path.append(str(chloro_file_folder))
sys.path.append(str(spectrum_file_folder))
# noinspection wrong-import-position
import analysis.chlorophyll_measurements
import analysis.spectrum_fitting
import analysis
