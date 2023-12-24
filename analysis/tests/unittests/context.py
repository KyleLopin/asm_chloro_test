# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
add the proper directory to python path for tests to work
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import pathlib
import sys

test_file_folder = pathlib.Path().absolute().parent.parent / "chlorophyll_measurements"
sys.path.append(str(test_file_folder))
# noinspection wrong-import-position
import analysis.chlorophyll_measurements
