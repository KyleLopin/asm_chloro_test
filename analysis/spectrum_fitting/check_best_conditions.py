# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Look through the metrics of the AS7262, AS7263 and AS7265x sensor for different
led, led current, and integration time
"""

__author__ = "Kyle Vitautas Lopin"

# for testing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
MODEL = "linear regression"
LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]


def read_data(sensor: str = "as7263", score_type="r2",
              leaf: str = "mango") -> dict:
    """ Get the data from the excel files that have the mean absolute error and
    r2 scores of the sensors and return a dictionary with the scores and labels of
    the conditions

    Args:
        sensor (str): sensor metrics to plot, only as7262 and as7263 works currently
        score_type (str): which metric to use, "mean_error" for the mean absolute errors
        or "r2" for the r2 scores
        leaf (str): what leaf to read the data for

    Returns:
        dict: A dictionary containing processed data with the following keys:
            - 'labels': A list of labels of the conditions used to collect the data.
            - 'training': average of the training scores
            - 'training errors': standard deviation of the training scores
            - 'test': average of the test scores
            - 'test errors': standard deviation of the test scores

    """
    filename = f"{sensor}_{score_type}.xlsx"
    excel_data = pd.ExcelFile(filename)
    training = []
    training_err = []
    test = []
    test_err = []
    labels = []
    for led_current in LED_CURRENTS:
        for int_time in INT_TIMES:
            sheet_name = f"sheet {int_time} msec {led_current}"
            data = excel_data.parse(sheet_name, index_col=0)
            data_pt = data.loc[leaf, MODEL]
            labels.append(f"{int_time} msec {led_current}")
            # data_splits = re.split('\u00b1,', data_pt)
            data_splits = data_pt.replace('\u00b1', ' ').replace(',', ' ').split()
            training.append(float(data_splits[0]))
            training_err.append(float(data_splits[1]))
            test.append(float(data_splits[2]))
            test_err.append(float(data_splits[3]))
    return {"labels": labels, "training": training, "training errors": training_err,
            "test": test, "test errors": test_err}


def anove_test(sensor="as7262"):
    # read data file first
    filename = f"{sensor}_anova.xlsx"
    excel_data = pd.ExcelFile(filename)
    for fruit in ["mango"]:
        data = excel_data.parse(fruit, index_col=0)
        samples = []
        for column in data.columns:
            samples.append(data[column].to_list())
        print(samples)
        print(*samples)
        anova_results = f_oneway(*samples)
        print(anova_results)


def read_data_df():  # refactor read_data for multi-bar plot
    test_df = pd.DataFrame(index=LED_CURRENTS,
                           columns=pd.Index(INT_TIMES, name=r"R^2"))
    test_err = pd.DataFrame(index=LED_CURRENTS,
                            columns=pd.Index(INT_TIMES, name=r"R^2"))
    filename = f"as7262_r2.xlsx"
    excel_data = pd.ExcelFile(filename)
    for led_current in LED_CURRENTS:
        for int_time in INT_TIMES:
            sheet_name = f"sheet {int_time} msec {led_current}"
            data = excel_data.parse(sheet_name, index_col=0)
            print(data)
            data_splits = data.loc["mango", MODEL].replace('\u00b1', ' ').replace(',', ' ').split()
            print(data_splits)
            test_df.loc[led_current, int_time] = float(data_splits[2])
            test_err.loc[led_current, int_time] = float(data_splits[3])
    print(test_df)
    test_df.plot(kind='bar', yerr=test_err)
    # sns.stripplot(data)
    plt.ylim([.6, 1.0])
    plt.xticks(rotation=0)
    print(test_df.mean())
    plt.show()


def plot_conditions(sensor="as7262", score_type="r2"):
    """ Make a bar plot for the scoring metrics of a sensor on based on its condition, i.e.
    integration time, and current.  Currently only for as7262 and as7263 sensors.

    Uses the read_data function to get the data.

    Args:
        sensor (str): sensor metrics to plot, only as7262 and as7263 works currently
        score_type (str): which metric to use, "mean_error" for the mean absolute errors
        or "r2" for the r2 scores

    Returns:
        None, plots the data

    """
    data = read_data(sensor=sensor, score_type=score_type)
    print(data["labels"])
    # get max condition
    max_idx = np.argmax(data["test"])
    print(f"best condition {data['labels'][max_idx]}")
    plt.bar(np.arange(0, len(data["labels"]), 1),
            data["test"], align='edge', yerr=data["test errors"],
            tick_label=data["labels"])
    plt.xticks(rotation=60)
    plt.ylim([0.7, 1.0])
    plt.tight_layout()
    plt.show()


def main_check_best_condition():
    # Read an excel file of filename <sensor>_mea.xlsx that has sheetnames
    # of the different leaf types and columns with a head of the different
    # integration time and led currents and values of individual cross
    # validated test scores for each leaf make a graph of a
    # scatter bar plot with
    # read data
    for sensor in ["as7262", "as7263"]:
        filename = f"{sensor}_anova.xlsx"
        excel_data = pd.ExcelFile(filename)
        for fruit in ["mango"]:
            data = excel_data.parse(fruit, index_col=0)
            # make a list of list to pass to the oneway-anova function as a pointer
            samples = []
            for column in data.columns:
                samples.append(data[column].to_list())
            anova_results = f_oneway(*samples)
        tips = sns.load_dataset("tips")
        print(tips)
        sns.stripplot(data)
        plt.show()


if __name__ == '__main__':
    # for leaf in ALL_LEAVES:
    #     print(f"leaf: {leaf}")
    #     plot_conditions()
    read_data_df()
    # anove_test()
    # main_check_best_condition()
