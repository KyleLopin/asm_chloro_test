# Copyright (c) 2023 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Look through the metrics of the AS7262, AS7263 and AS7265x sensor for different
led, led current, and integration time
"""

__author__ = "Kyle Vitautas Lopin"

# installed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import alexandergovern, f_oneway, kruskal, ttest_ind
MODEL = "linear regression"
LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
plt.style.use("seaborn-v0_8-darkgrid")


def read_data_depr(sensor: str = "as7263", score_type="r2",
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


def anove_test(_df:pd.DataFrame) -> float:
    """ Calculate the one way ANOVA (Analysis of variance) of a DataFrame

    Take a DataFrame with different conditions for each column and calculate the
    one way ANOVA (Analysis of variance) to test if the conditions are
    statistically different.  Also performs the Kruskal-Wallis H-test and the
    Alexander-Govern approximation tests, which are similiar to ANOVA but
    do not assume the samples have the same variance, so they will give slightly
    different p-values than the ANOVA results.

    Args:
        _df (pd.DataFrame): DataFrame to perform the tests on

    Returns:
        float: p-value of the ANOVA test

    """
    samples = []  # holder for different conditions results, a list of lists
    # go through each column and save the measurements to a list to pass to the
    # scipy f_oneway function
    for column in _df.columns:
        samples.append(_df[column].to_list())
    anova_results = f_oneway(*samples)  # pass the list of list as a pointer to unpack
    kruskal_results = kruskal(*samples)
    alexa_results = alexandergovern(*samples)
    print(kruskal_results)
    print(alexa_results)
    return anova_results.pvalue


def anova_test_all_currents_grouped(_df:pd.DataFrame):
    """ Analyze samples from a DataFrame for different LED current levels.

    This function aggregates samples from the DataFrame based on LED current levels
    and performs statistical analysis including one-way ANOVA, Kruskal-Wallis H test,
    and Alexander-Govern test. It also identifies the LED current level with the highest
    mean value and performs pairwise t-tests between the samples.

    Args:
        _df (pd.DataFrame): The DataFrame containing the samples.

    Returns:
        None: just prints the results out

    """
    samples = []
    for _ in LED_CURRENTS:
        samples.append([])  # add an empty list for each current
    for column in _df.columns:
        # for each column, check the led current used
        for i, led_current in enumerate(LED_CURRENTS):
            if led_current in column:  # and add it to the correct list
                samples[i].extend(_df[column].to_list())
    anova_results = f_oneway(*samples)  # pass the list of list as a pointer to unpack
    kruskal_results = kruskal(*samples)
    alexa_results = alexandergovern(*samples)
    print("current statistics")
    print(anova_results)
    print(kruskal_results)
    print(alexa_results)
    # find best samples
    means = np.mean(np.array(samples), axis=1)
    max_idx = np.argmax(means)
    t_scores = t_tests_from_samples(samples[max_idx], samples)
    # print the conditions and the scores
    print(LED_CURRENTS)
    print(t_scores)


def anova_int_times(_df:pd.DataFrame):
    """ Analyze samples from a DataFrame for different integration times.

    This function aggregates samples from the DataFrame based on integration times and
    performs statistical analysis including one-way ANOVA, Kruskal-Wallis H test,
    and Alexander-Govern test. It also identifies the integration time with the highest mean
    value and performs pairwise t-tests between the samples.

    Args:
        _df (pd.DataFrame): The DataFrame containing the samples.

    Returns:
        None: just prints the results out

    """

    samples = []
    for _ in INT_TIMES:
        samples.append([])  # add an empty list for each current
    for column in _df.columns:
        for i, int_time in enumerate(INT_TIMES):
            if column.startswith(f"{int_time} msec"):
                samples[i].extend(_df[column].to_list())
    anova_results = f_oneway(*samples)  # pass the list of list as a pointer to unpack
    kruskal_results = kruskal(*samples)
    alexa_results = alexandergovern(*samples)
    print("integration time statistics")
    print(anova_results)
    print(kruskal_results)
    print(alexa_results)
    # find best samples
    means = np.mean(np.array(samples), axis=1)
    print(INT_TIMES)
    max_idx = np.argmax(means)
    t_scores = t_tests_from_samples(samples[max_idx], samples)
    print(t_scores)


def make_pg_anova_table(sensor: str="as7262", score_type: str = "r2"
                        ) -> pd.DataFrame:
    """ Make a DataFrame that can be passed into a pingouin anova model.

    Read the sensor data from read_data_df function for all leaf types
    and then format it so that a pingouin model can use it

    sensor (str): sensor metrics to plot, only as7262 and as7263 works currently
        score_type (str): which metric to use, "mean_error" for the mean absolute errors
        or "r2" for the r2 scores

    Returns:
        pd.DataFrame: columns of the different conditions and the cross
        validated scores.

    """
    final_df = pd.DataFrame()
    for leaf in ALL_LEAVES:
        scores_df, _ = read_data_df(sensor=sensor, score_type=score_type,
                                    leaf=leaf)
        scores_df = scores_df.stack(level=0).reset_index()
        final_df = pd.concat([final_df, scores_df], ignore_index=True)
    final_df = final_df.rename(columns={"level_0": "current",
                                        "R^2": "int time",
                                        0: "r2"})
    final_df['r2'] = pd.to_numeric(final_df['r2'])
    return final_df


def test_factorial_anova(_df: pd):
    """ Run a 2-factor ANOVA test on the sensor

    Args:
        _df:

    Returns:

    """
    pg_df = make_pg_anova_table(sensor="as7262", score_type="r2")
    model1 = pg.anova(dv='r2', between=['current', 'int time'],
                      data=pg_df, detailed=True)
    round(model1, 4)
    print(model1)


def read_data_df(sensor: str="as7262", score_type: str = "r2",
                 leaf: str="mango") -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Read the cross validation test scores from the saved files and return
    the means and standard deviations of the linear regression runs.

    Args:
        sensor (str):
        score_type (str):
        leaf (str):

    Returns:
        pd.DataFrame: DataFrame with test scores with an index of the LED currents used
        and columns of the integration times
        pd.DataFrame: DataFrame with standar deviation of the test scores with an
        index of the LED currents used and columns of the integration times

    """
    # refactor read_data for multi-bar plot
    test_df = pd.DataFrame(index=LED_CURRENTS,
                           columns=pd.Index(INT_TIMES, name=r"R^2"))
    test_err = pd.DataFrame(index=LED_CURRENTS,
                            columns=pd.Index(INT_TIMES, name=r"R^2"))
    filename = f"{sensor}_{score_type}.xlsx"
    excel_data = pd.ExcelFile(filename)
    # the files with have sheet names for each integration time and led current
    # go through all combinations to get them
    for led_current in LED_CURRENTS:
        for int_time in INT_TIMES:
            sheet_name = f"sheet {int_time} msec {led_current}"
            data = excel_data.parse(sheet_name, index_col=0)
            # data is in the form training mean +- training std, test mean += test std
            data_splits = data.loc[leaf, MODEL].replace('\u00b1', ' ').replace(',', ' ').split()
            test_df.loc[led_current, int_time] = float(data_splits[2])
            test_err.loc[led_current, int_time] = float(data_splits[3])
    return test_df, test_err


def t_tests_from_samples(best_sample:list[float],
                         rest_of_samples:list[list[float]]) -> list:
    """ Do a pairwise t-test over a set of samples.

    Take a list of scores to compare (should be the highest scoring sample) to
    a list of lists for the rest of the samples to calculate the p-values for.
    Returns a list of each p-value in the same order as the list of list passed in

    Args:
        best_sample (list[float]): A single list of values to compare to all the rest
        rest_of_samples (list[list[float]]): List of lists of values to compare
        to the first argument

    Returns:
        list: p-values for each list in the rest_of_samples argument

    """
    p_values = []
    for sample in rest_of_samples:
        _, p = ttest_ind(best_sample, sample)
        p_values.append(p)
    return p_values


def main_check_best_condition(sensor="as7262", score_type="r2"):
    """ Plot the average of each leafs cross-validation error.

    Args:
        sensor (str): sensor averages to plot, only as7262 and as7263 works currently
        score_type (str) : which score type to use, r2 or mean_error

    Returns:
        None: plots a graph of the best conditions

    """
    # store all the leafs scores to get the mean and std from
    summary = pd.DataFrame()
    # summary will have columns with the conditions i.e. 50 msec 12.5 mA and row
    # for each leaf, but indexed 0, 1, .. 5
    for i, leaf in enumerate(ALL_LEAVES):
        scores_df, _ = read_data_df(sensor=sensor, score_type=score_type,
                                    leaf=leaf)
        for _column in scores_df.columns:
            for row in scores_df.index:
                summary.loc[i, f"{_column} msec {row}"] = scores_df.loc[row, _column]
    anova_p_value = anove_test(summary)
    # make the mean and standard deviations DataFrames from summary
    _means = summary.mean()
    _stds = summary.std()
    max_idx = _means.idxmax()
    max_dist = summary[max_idx]
    t_tests = []
    for _column in summary.columns:
        _, p = ttest_ind(summary[_column], max_dist)
        t_tests.append([p, _column])
    # make the dataFrame that plot will make correctly with each current grouped together
    # and the integration times color coded
    means = pd.DataFrame()
    stds = pd.DataFrame()
    for int_time in INT_TIMES:
        for current in LED_CURRENTS:
            means.loc[current, int_time] = _means[f"{int_time} msec {current}"]
            stds.loc[current, int_time] = _stds[f"{int_time} msec {current}"]

    ax = means.plot(kind='bar', yerr=stds)
    for t_test in t_tests:
        print(t_test)
    print(f"ANOVA p value: {anova_p_value}")
    anova_test_all_currents_grouped(summary)
    anova_int_times(summary)
    # ax = plt.gca()
    print(ax.get_facecolor())
    # ax.set_facecolor([.84, .84, .90, 1.0])
    # plt.axhline(y=summary[max_idx].mean(), color='white')
    # ax.set_yticks([summary[max_idx].mean()], minor=True)
    # ax.yaxis.grid(True, which='minor')
    print(summary[max_idx].mean())
    plt.show()


if __name__ == '__main__':
    main_check_best_condition()
    test_factorial_anova(1)
