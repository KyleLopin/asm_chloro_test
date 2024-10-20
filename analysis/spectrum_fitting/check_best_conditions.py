# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make an Anova tabe for AS7262, AS7263 and AS7265x sensor for different
led current, measurement types, and integration times for the different leave types
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from collections import defaultdict
import itertools
import warnings

# installed libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ARDRegression, HuberRegressor, LassoLarsIC, LinearRegression
from sklearn.model_selection import cross_val_score, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# local files
import get_data

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
MEASUREMENT_TYPES = ["raw", "reflectance", "absorbance"]
SENSORS = ["as7262", "as7263", "as7265x"]
PREPROCESS = ["Poly", "No Poly"]
SCORE = "r2"

regression_models_2_3 = {
    "ARD": ARDRegression(lambda_2=0.001),
    "Huber Regression": HuberRegressor(max_iter=10000),
    "Lasso IC": LassoLarsIC(criterion='bic'),
    "Linear Regression": LinearRegression(),
    "PLS": PLSRegression(n_components=10)
}

regression_models_5x = {
    "ARD": ARDRegression(lambda_2=0.01),
    "Huber Regression": HuberRegressor(max_iter=10000),
    "Lasso IC": LassoLarsIC(criterion='bic'),
    "Linear Regression": LinearRegression(),
    "PLS": PLSRegression(n_components=20)
}

cvs = {
    "Shuffle": GroupShuffleSplit(n_splits=5, test_size=0.2),
    "K Fold": GroupKFold(n_splits=5)
}


def make_anova_table_files():
    """ make .csv file for each sensor and leaf to run ANOVA analysis on"""
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            make_anova_table_file(leaf, sensor)


def make_anova_table_file(leaf: str, sensor: str):
    """
    Generate an ANOVA table file for a given leaf and sensor combination by
    running a set of regression models with various parameter combinations.

    This function performs the following steps:
    1. Defines a list of parameter combinations based on the sensor type.
    2. Retrieves the feature matrix (X), target variable (Y), and group labels for cross-validation.
    3. Applies data preprocessing, including polynomial feature transformation and scaling.
    4. Iterates over different regression models and cross-validation techniques.
    5. Evaluates the models using cross-validation and collects the results.
    6. Saves the results to a CSV file in the "ANOVA_data" directory.

    Parameters:
    leaf (str): The name of the leaf sample being analyzed.
    sensor (str): The sensor type used for data collection.

    The output CSV file will be named "ANOVA_<leaf>_<sensor>.csv" and will contain
    the details of each parameter combination, including the regression model,
    cross-validation technique, and score.
    """
    print(f"making {leaf} {sensor} ANOVA table")
    results = []
    int_times = INT_TIMES
    if sensor == "as7265x":
        int_times = [50, 100, 150]
    combinations = itertools.product(MEASUREMENT_TYPES, int_times, LED_CURRENTS, PREPROCESS)

    for measure_type, int_time, current, preprocess in combinations:
        x, y, groups = get_data.get_x_y(
            leaf=leaf, sensor=sensor,
            measurement_type=measure_type,
            int_time=int_time, led_current=current,
            send_leaf_numbers=True)
        y = y["Avg Total Chlorophyll (µg/cm2)"]
        regrs = regression_models_2_3
        if sensor == "as7265x":  # change if as7265x sensor
            regrs = regression_models_5x
        if preprocess == "Poly":
            x = PolynomialFeatures(degree=2).fit_transform(x)
        else:
            if sensor in ["as7262", "as7263"]:
                regrs["PLS"] = PLSRegression(n_components=6)
            elif sensor == "as7265x":
                regrs["PLS"] = PLSRegression(n_components=18)
        x = StandardScaler().fit_transform(x)

        for regr_name, regr in regrs.items():
            for cv_name, cv in cvs.items():
                # get 5 test scores for each combo
                scores = cross_val_score(
                    regr, x, y, groups=groups, scoring=SCORE,
                    cv=cv)
                # Append the results to the list
                for score in scores:
                    results.append({
                        "Leaf": leaf,
                        "Sensor": sensor,
                        "Measurement Type": measure_type,
                        "Integration Time": int_time,
                        "LED Current": current,
                        "Preprocess": preprocess,
                        "Regression Model": regr_name,
                        "Cross Validation": cv_name,
                        "Score": score
                    })

                # Create a DataFrame from the results list
            results_df = pd.DataFrame(results)

            # Save the DataFrame to a CSV file
            filename = f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv"
            results_df.to_csv(filename, index=False)


def print_pg_anova_tables():
    """ Print out the ANOVA tables for each leaf, sensor combintation """
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            print_pg_anova_table(leaf, sensor)


def print_pg_anova_table(leaf: str, sensor: str):
    """
    Filter and save the ANOVA results for a given leaf and sensor combination.

    This function performs the following steps:
    1. Reads the ANOVA results from a CSV file named "ANOVA_<leaf>_<sensor>.csv"
       located in the "ANOVA_data" directory.
    2. Conducts an ANOVA test on the 'Score' variable with multiple factors:
       'Measurement Type', 'Integration Time', 'LED Current', 'Preprocess',
       'Regression Model', and 'Cross Validation'.
    3. Applies the Bonferroni correction to the p-values to account for
       multiple comparisons.
    4. Filters the ANOVA results to include only rows where the corrected
       p-value ('p-corrected') is less than 0.001.
    5. Prints the filtered ANOVA results with all rows displayed.
    6. Saves the filtered results to a new CSV file named
       "Filtered_ANOVA_<leaf>_<sensor>.csv" in the "ANOVA_data" directory.

    Parameters:
    leaf (str): The name of the leaf sample being analyzed.
    sensor (str): The sensor type used for data collection.

    The output file will contain the filtered ANOVA results based on the
    significance level specified for the Bonferroni-corrected p-values.
    """
    print(f"{leaf} {sensor}")
    filename = f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv"
    df = pd.read_csv(filename)

    # Perform the ANOVA test
    aov = pg.anova(dv='Score',
                   between=['Measurement Type', 'Integration Time', 'LED Current',
                            'Preprocess', 'Regression Model', 'Cross Validation'],
                   data=df)

    # Apply Bonferroni correction to the p-values
    aov['p-corrected'] = pg.multicomp(aov['p-unc'], method='bonferroni')[1]

    # Filter rows where p-corrected > 0.001
    filtered_aov = aov[aov['p-corrected'] < 0.001]

    pd.set_option('display.max_rows', None)  # Ensure all rows are displayed
    print(filtered_aov)

    # Save the filtered ANOVA table to a new CSV file
    output_filename = f"ANOVA_data/Filtered_ANOVA_{leaf}_{sensor}.csv"
    filtered_aov.to_csv(output_filename, index=False)


def plot_scores_for_leaf_sensor(df, leaf, sensor, pdf=None):
    """
    Plot the best scores for a given leaf and sensor based on the maximum score
    achieved across different conditions.

    This function performs the following steps:
    1. Filters the provided DataFrame to include only rows corresponding to the
       specified leaf and sensor.
    2. Groups the filtered DataFrame by a specified column ('Cross Validation')
       and calculates the maximum 'Score' for each group.
    3. Plots a bar chart of the maximum scores for the specified leaf and sensor,
       with the scores on the y-axis and the group labels on the x-axis.
    4. Optionally saves the plot to a PDF file if a PDF object is provided;
       otherwise, displays the plot.

    Parameters:
    leaf (str): The name of the leaf sample being analyzed.
    sensor (str): The sensor type used for data collection.
    df (pd.DataFrame): The DataFrame containing the ANOVA results.
    pdf (PdfPages, optional): A PdfPages object to save the plot to a PDF file.
                              If not provided, the plot will be displayed.

    The plot displays the best scores achieved for different 'Cross Validation'
    conditions, helping to identify which models perform the best for the given
    leaf and sensor.
    """
    # Filter the DataFrame for the given leaf and sensor
    df_filtered = df[(df['Leaf'] == leaf) & (df['Sensor'] == sensor)]

    # conditions_that can be plotted: ['LED Current', 'Integration Time',
    # 'Measurement Type', 'Preprocess', 'Cross Validation']
    column = 'Cross Validation'
    # Group by 'Regression Model' and calculate the max of 'Score'
    grouped = df_filtered.groupby(column).agg({'Score': 'max'}).reset_index()
    print()
    # Plot the max score
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.bar(grouped[column], grouped['Score'], color='skyblue')
    plt.title(f'Best Scores for {leaf} - {sensor}')
    plt.xlabel('Regression Model')
    plt.ylabel('Best Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.ylim([0.8, 1.0])

    if pdf:
        pdf.savefig(fig)
    else:
        plt.show()


def collect_best_scores_for_each_model():
    # Dictionary to hold lists of best scores for each model
    model_best_scores = defaultdict(list)

    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            # Load the data for the current leaf/sensor combination
            df = pd.read_csv(f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv")

            # Group by 'Regression Model' and calculate the max of 'Score'
            grouped = df.groupby('Regression Model').agg({'Score': 'max'}).reset_index()

            # Store the best score for each model
            for index, row in grouped.iterrows():
                model = row['Regression Model']
                best_score = row['Score']
                model_best_scores[model].append(best_score)

    return model_best_scores


def plot_model_performance(model_best_scores):
    # Prepare lists for plotting
    models = []
    mean_scores = []
    std_scores = []

    # Calculate mean and std for each model
    for model, scores in model_best_scores.items():
        models.append(model)
        mean_scores.append(pd.Series(scores).mean())
        std_scores.append(pd.Series(scores).std())

    # Plotting the mean and std of the best scores for each model
    plt.figure(figsize=(10, 6))
    plt.errorbar(models, mean_scores, yerr=std_scores, fmt='o', capsize=5, elinewidth=2,
                 markeredgewidth=2)
    plt.title('Mean and Std of Best Scores for Each Regression Model')
    plt.xlabel('Regression Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_combined_anova_tables():
    combined_data = []
    # make combined dataset
    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            df = pd.read_csv(f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv")
            df['Leaf'] = leaf
            df['Sensor'] = sensor
            combined_data.append(df)
    combined_df = pd.concat(combined_data, ignore_index=True)
    # Convert relevant columns to float32
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[numeric_cols] = combined_df[numeric_cols].astype(np.float32)
    return combined_df

def print_1_way_anova(ind_variable):
    print("check")
    df = get_combined_anova_tables()
    print(df.shape)
    print(df)
    if True:
        df = df[df['Cross Validation'] == "Shuffle"]
        df = df[df['Preprocess'] == "Poly"]
    print(df.shape)
    all_variable = ['Leaf', 'Sensor', 'Regression Model',
                    'Cross Validation', 'Preprocess',
                    "LED Current", "Integration Time",
                    "Measurement Type"]
    # aov = pg.anova(dv='Score',
    #                between=[ind_variable],
    #                data=df,
    #                detailed=True)

    aov = pg.anova(dv='Score',
                   between=['Leaf', 'Sensor', 'Regression Model',
                            "LED Current", "Integration Time",
                            "Measurement Type"],
                   data=df,
                   detailed=True)
    # Display the ANOVA table
    print(aov)


def paiwise_ttests():
    """ Run and print """
    df = get_combined_anova_tables()
    # Run pairwise t-tests for 'Leaf'
    ttest_leaf = pg.pairwise_ttests(dv='Score', between='Leaf', data=df, padjust='bonf')

    # Run pairwise t-tests for 'Sensor'
    print(df["Sensor"].unique())
    ttest_sensor = pg.pairwise_ttests(dv='Score', between='Sensor', data=df, padjust='bonf')

    # Run pairwise t-tests for 'Regression Model'
    ttest_reg_model = pg.pairwise_ttests(dv='Score', between='Regression Model', data=df,
                                         padjust='bonf')

    # Run pairwise t-tests for 'LED Current'
    ttest_led_current = pg.pairwise_ttests(dv='Score', between='LED Current', data=df,
                                           padjust='bonf')

    # Run pairwise t-tests for 'Integration Time'
    ttest_integration_time = pg.pairwise_ttests(dv='Score', between='Integration Time', data=df,
                                                padjust='bonf')

    # Run pairwise t-tests for 'Measurement Type'
    ttest_measurement_type = pg.pairwise_ttests(dv='Score', between='Measurement Type', data=df,
                                                padjust='bonf')

    # Display the results for each t-test
    print(ttest_leaf)
    print(ttest_sensor)
    print(ttest_reg_model)
    print(ttest_led_current)
    print(ttest_integration_time)
    print(ttest_measurement_type)


def get_best_conditions() -> dict:
    """
    Combines the functionality to find:
    1. The best conditions for each sensor across all leaves (best overall sensor conditions).
    2. The best overall conditions for all leaves and sensors combined.

    Returns:
    dict: A dictionary containing two DataFrames:
          - 'best_sensor_conditions': Best conditions for each sensor overall.
          - 'best_overall_conditions': Best overall conditions for all leaves and sensors combined.

    Example usage:
    --------------
    # Get both the best sensor conditions and the best overall conditions
    best_conditions_dict = get_best_conditions()

    # Print both results
    print("Best Overall Sensor Conditions:\n", best_conditions_dict['best_sensor_conditions'])
    print("\nBest Overall Conditions for All Leaves and Sensors:\n", best_conditions_dict['best_overall_conditions'])
    """
    # Get the DataFrame from the combined ANOVA tables
    df = get_combined_anova_tables()

    # --- Best Conditions for Each Sensor (Overall across all leaves) ---
    # Group by 'Sensor' and other conditions, calculate the average 'Score' across all leaves
    mean_scores_sensor = df.groupby(
        ['Sensor', 'Preprocess', 'Regression Model', 'Cross Validation', 'LED Current',
         'Integration Time', 'Measurement Type']
    )['Score'].mean().reset_index()

    # Find the row with the highest average score for each sensor
    best_avg_conditions_for_sensor = mean_scores_sensor.loc[
        mean_scores_sensor.groupby('Sensor')['Score'].idxmax()]

    # --- Best Overall Conditions for All Leaves and Sensors ---
    # Group by all condition columns without 'Sensor' and 'Leaf',
    # then calculate the mean score across all leaves and sensors
    mean_scores_overall = df.groupby(
        ['Preprocess', 'Regression Model', 'Cross Validation', 'LED Current', 'Integration Time',
         'Measurement Type']
    )['Score'].mean().reset_index()

    # Find the row with the highest overall score
    best_overall_conditions = mean_scores_overall.loc[mean_scores_overall['Score'].idxmax()]

    # Return both DataFrames as a dictionary
    return {
        'best_sensor_conditions': best_avg_conditions_for_sensor,
        'best_overall_conditions': best_overall_conditions
    }


def collect_best_scores_for_each_condition(condition_name):
    # Dictionary to hold lists of best scores for each combination
    # of the condition and regression models
    condition_best_scores = defaultdict(lambda: defaultdict(list))

    for leaf in ALL_LEAVES:
        for sensor in SENSORS:
            # Load the data for the current leaf/sensor combination
            df = pd.read_csv(f"ANOVA_data/ANOVA_{leaf}_{sensor}.csv")

            # Group by the condition and regression model,
            # then find the max score for each combination
            grouped = df.groupby([condition_name, 'Regression Model']
                                 ).agg({'Score': 'max'}).reset_index()

            # Store the best score for each regression model under each condition
            for index, row in grouped.iterrows():
                condition_value = row[condition_name]
                model = row['Regression Model']
                best_score = row['Score']
                condition_best_scores[condition_value][model].append(best_score)

    return condition_best_scores

def plot_performance_for_condition(condition_name, condition_best_scores):
    # Prepare the plot for each condition
    plt.figure(figsize=(10, 6))

    # Iterate through each condition value (e.g., each 'LED Current', 'Integration Time')
    for condition_value, model_scores in condition_best_scores.items():
        models = []
        mean_scores = []
        std_scores = []

        # Calculate mean and std for each regression model under the current condition value
        for model, scores in model_scores.items():
            models.append(model)
            mean_scores.append(pd.Series(scores).mean())
            std_scores.append(pd.Series(scores).std())

        # Plotting the mean and std of best scores for each regression model under the current condition
        plt.errorbar(models, mean_scores, yerr=std_scores, fmt='o', capsize=5, elinewidth=2, markeredgewidth=2,
                     label=f"{condition_name}: {condition_value}")

    plt.title(f'Mean and Std of Best Scores for Different {condition_name}')
    plt.xlabel('Regression Model')
    plt.ylabel('Score')
    plt.legend(title=f'{condition_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_and_check_statistics_for_conditions() -> None:
    """
    Plots bar graphs comparing the best overall conditions, best sensor conditions,
    and the maximum score for each leaf. Additionally, performs statistical
    significance tests (paired t-test) for three comparisons:
    - Best leaf scores vs. best overall condition scores
    - Best leaf scores vs. best sensor condition scores
    - Best sensor scores vs. best overall condition scores

    The function will:
    - Plot bar graphs for each sensor showing the scores for the best overall condition,
      best sensor-specific condition, and the maximum score for each leaf.
    - Print out the detailed conditions and statistical test results for each leaf.
    """
    # Get the DataFrame from combined ANOVA tables
    df = get_combined_anova_tables()

    # Get the best overall conditions and sensor-specific conditions
    best_conditions = get_best_conditions()
    best_sensor_conditions = best_conditions['best_sensor_conditions']
    best_overall_conditions = best_conditions['best_overall_conditions']

    # Inner function to filter data based on conditions
    def filter_data(leaf_data, condition):
        return (leaf_data['Regression Model'] == condition['Regression Model']) & \
            (leaf_data['LED Current'] == condition['LED Current']) & \
            (leaf_data['Integration Time'] == condition['Integration Time']) & \
            (leaf_data['Measurement Type'] == condition['Measurement Type']) & \
            (leaf_data['Preprocess'] == condition['Preprocess']) & \
            (leaf_data['Cross Validation'] == condition['Cross Validation'])

    # List of sensors
    sensors = df['Sensor'].unique()

    # Loop through each sensor and plot a comparison graph for the sensor across all leaves
    for sensor in sensors:
        plt.figure(figsize=(12, 8))

        # Initialize lists to store data for plotting
        leaves = []
        best_overall_scores = []
        best_leaf_scores = []
        max_leaf_scores = []
        best_overall_std = []
        best_leaf_std = []
        max_leaf_std = []

        # Print the best overall conditions used for this sensor
        print(f"\n=== Sensor: {sensor} ===")
        print("Best Overall Condition:")
        print(best_overall_conditions)

        # Print the best sensor-specific condition for this sensor
        sensor_condition = best_sensor_conditions[best_sensor_conditions['Sensor'] == sensor].iloc[0]
        print("Best Sensor Condition:")
        print(sensor_condition)

        # Get scores for each leaf
        for leaf in df['Leaf'].unique():
            # Filter for the current leaf and sensor combination
            leaf_data = df[(df['Leaf'] == leaf) & (df['Sensor'] == sensor)]

            # --- Best Overall Conditions ---
            best_overall_cond = leaf_data[filter_data(leaf_data, best_overall_conditions)]['Score'].mean()
            best_overall_cond_std = leaf_data[filter_data(leaf_data, best_overall_conditions)]['Score'].std()

            # --- Best Sensor Conditions ---
            best_sensor_cond = leaf_data[filter_data(leaf_data, sensor_condition)]['Score'].mean()
            best_sensor_cond_std = leaf_data[filter_data(leaf_data, sensor_condition)]['Score'].std()

            # --- Best Leaf Condition (Max Score for the Leaf) ---
            best_leaf_cond = leaf_data.loc[leaf_data['Score'].idxmax()]
            best_leaf_cond_score = best_leaf_cond['Score']

            # Print the detailed conditions
            print(f"\nLeaf: {leaf}")
            print(f"Best Leaf Condition:")
            print(best_leaf_cond.tolist())  # This prints the entire row including all relevant condition parameters

            # Perform paired t-tests for statistical significance
            comparison_data = pd.DataFrame({
                'Best_Leaf_Score': leaf_data['Score'],
                'Best_Overall_Score': [best_overall_cond] * len(leaf_data),
                'Best_Sensor_Score': [best_sensor_cond] * len(leaf_data)
            })

            # T-test: Best leaf vs. best overall
            ttest_leaf_vs_overall = pg.ttest(comparison_data['Best_Leaf_Score'], comparison_data['Best_Overall_Score'], paired=True)
            t_statistic_leaf_vs_overall = ttest_leaf_vs_overall['T'].iloc[0]
            p_value_leaf_vs_overall = ttest_leaf_vs_overall['p-val'].iloc[0]

            # T-test: Best leaf vs. best sensor
            ttest_leaf_vs_sensor = pg.ttest(comparison_data['Best_Leaf_Score'], comparison_data['Best_Sensor_Score'], paired=True)
            t_statistic_leaf_vs_sensor = ttest_leaf_vs_sensor['T'].iloc[0]
            p_value_leaf_vs_sensor = ttest_leaf_vs_sensor['p-val'].iloc[0]

            # T-test: Best sensor vs. best overall
            ttest_sensor_vs_overall = pg.ttest(comparison_data['Best_Sensor_Score'], comparison_data['Best_Overall_Score'], paired=True)
            t_statistic_sensor_vs_overall = ttest_sensor_vs_overall['T'].iloc[0]
            p_value_sensor_vs_overall = ttest_sensor_vs_overall['p-val'].iloc[0]

            # Print the statistical test results
            print(f"Statistical Tests for Leaf: {leaf}")
            print(f"Leaf vs. Overall - T-Statistic: {t_statistic_leaf_vs_overall}, P-Value: {p_value_leaf_vs_overall}")
            print(f"Leaf vs. Sensor - T-Statistic: {t_statistic_leaf_vs_sensor}, P-Value: {p_value_leaf_vs_sensor}")
            print(f"Sensor vs. Overall - T-Statistic: {t_statistic_sensor_vs_overall}, P-Value: {p_value_sensor_vs_overall}")

            # Append data for plotting
            leaves.append(leaf)
            best_overall_scores.append(best_overall_cond)
            best_leaf_scores.append(best_sensor_cond)
            max_leaf_scores.append(best_leaf_cond_score)
            best_overall_std.append(best_overall_cond_std)
            best_leaf_std.append(best_sensor_cond_std)
            max_leaf_std.append(leaf_data['Score'].std())

        # Plot the bar graph
        x = range(len(leaves))
        bar_width = 0.2

        bars1 = plt.bar(x, best_overall_scores, width=bar_width, label='Best Overall Conditions',
                        align='center', yerr=best_overall_std, capsize=5)
        bars2 = plt.bar([p + bar_width for p in x], best_leaf_scores, width=bar_width,
                        label=f'Best {sensor} Conditions', align='center', yerr=best_leaf_std, capsize=5)
        bars3 = plt.bar([p + 2 * bar_width for p in x], max_leaf_scores, width=bar_width,
                        label='Max Leaf Condition', align='center', yerr=max_leaf_std, capsize=5)

        # Add labels and title
        plt.xticks([p + bar_width for p in x], leaves)
        plt.ylabel('Score')
        plt.title(f'Comparison of Best Conditions for {sensor} across Leaves')
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.ylim([0.5, 1])
        plt.show()


def check_statistical_significance(df: pd.DataFrame) -> None:
    """
    Checks the statistical significance between the best leaf scores and the best overall condition scores
    using Pingouin's t-test for each sensor.

    Parameters:
    df (pd.DataFrame): DataFrame containing the scores and conditions.

    Returns:
    None
    """
    # Get the best conditions (both sensor and overall)
    best_conditions = get_best_conditions()

    # Extract the best overall condition from the returned dictionary
    best_overall_condition = best_conditions['best_overall_conditions']

    # Loop through each sensor
    for sensor in df['Sensor'].unique():
        print(f"\n=== Sensor: {sensor} ===")

        # Filter the data for the current sensor
        sensor_data = df[df['Sensor'] == sensor]

        # Loop through each leaf for the current sensor
        for leaf in sensor_data['Leaf'].unique():
            # Filter the data for the current leaf
            leaf_data = sensor_data[sensor_data['Leaf'] == leaf]

            # Get the best score for the current leaf
            best_leaf_score = leaf_data['Score'].max()

            # Get the score for the best overall condition for comparison
            # Assuming 'Score' is the column in the DataFrame returned for best overall condition
            best_overall_score = best_overall_condition['Score'].mean()

            # Prepare data for the paired t-test
            # Create a DataFrame for pingouin t-test
            comparison_data = pd.DataFrame({
                'Best_Leaf_Score': leaf_data['Score'],
                'Best_Overall_Score': [best_overall_score] * len(leaf_data)
                # Repeat the best overall score
            })

            # Perform paired t-test using pingouin
            ttest_results = pg.ttest(comparison_data['Best_Leaf_Score'],
                                     comparison_data['Best_Overall_Score'], paired=True)

            # Extract the results
            t_statistic = ttest_results['T'].iloc[0]
            p_value = ttest_results['p-val'].iloc[0]

            # Print the results
            print(
                f"Leaf: {leaf}, Best Leaf Score: {best_leaf_score}, Best Overall Condition Score: {best_overall_score}")
            print(f"T-Statistic: {t_statistic}, P-Value: {p_value}\n")


def find_best_conditions_for_led_and_integration_time() -> None:
    """
    Finds the best LED current and integration time combinations for each leaf and sensor
    based on the mean score and performs statistical tests comparing:
    1. Best overall conditions
    2. Best sensor-specific conditions
    3. Best leaf-specific conditions
    """

    # Get the DataFrame from combined ANOVA tables
    df = get_combined_anova_tables()

    # Get the best overall conditions and sensor-specific conditions
    best_conditions = get_best_conditions()
    best_overall_conditions = best_conditions['best_overall_conditions']
    best_sensor_conditions = best_conditions['best_sensor_conditions']

    # List of sensors
    sensors = df['Sensor'].unique()

    # Loop through each sensor
    for sensor in sensors:
        plt.figure(figsize=(12, 8))

        # Initialize lists to store data for plotting
        leaves = []
        best_overall_scores = []
        best_sensor_scores = []
        best_leaf_scores = []
        best_overall_std = []
        best_sensor_std = []
        best_leaf_std = []

        # Print the sensor details
        print(f"\n=== Sensor: {sensor} ===")

        # Extract the best sensor-specific condition for this sensor
        sensor_condition = best_sensor_conditions[best_sensor_conditions['Sensor'] == sensor].iloc[0]

        # Loop through each leaf
        for leaf in df['Leaf'].unique():
            # Filter data for the current leaf and sensor
            leaf_data = df[(df['Leaf'] == leaf) & (df['Sensor'] == sensor)]

            # Group by LED Current and Integration Time and calculate the mean score
            grouped = leaf_data.groupby(['LED Current', 'Integration Time'])['Score'].mean().reset_index()

            # --- Best Leaf Condition ---
            # Find the combination with the highest mean score
            best_leaf_combination = grouped.loc[grouped['Score'].idxmax()]
            best_leaf_led = best_leaf_combination['LED Current']
            best_leaf_integration = best_leaf_combination['Integration Time']
            best_leaf_score = best_leaf_combination['Score']

            # --- Best Overall Conditions ---
            # Get the score for the best overall condition
            best_overall_cond = leaf_data[
                (leaf_data['LED Current'] == best_overall_conditions['LED Current']) &
                (leaf_data['Integration Time'] == best_overall_conditions['Integration Time'])
            ]['Score'].mean()
            best_overall_cond_std = leaf_data[
                (leaf_data['LED Current'] == best_overall_conditions['LED Current']) &
                (leaf_data['Integration Time'] == best_overall_conditions['Integration Time'])
            ]['Score'].std()

            # --- Best Sensor Conditions ---
            # Get the score for the best sensor condition
            best_sensor_cond = leaf_data[
                (leaf_data['LED Current'] == sensor_condition['LED Current']) &
                (leaf_data['Integration Time'] == sensor_condition['Integration Time'])
            ]['Score'].mean()
            best_sensor_cond_std = leaf_data[
                (leaf_data['LED Current'] == sensor_condition['LED Current']) &
                (leaf_data['Integration Time'] == sensor_condition['Integration Time'])
            ]['Score'].std()

            # Print details for the current leaf
            print(f"\nLeaf: {leaf}")
            print(f"Best Leaf Condition: LED Current = {best_leaf_led}, Integration Time = {best_leaf_integration}, Score = {best_leaf_score}")
            print(f"Best Overall Score: {best_overall_cond}")
            print(f"Best Sensor Score: {best_sensor_cond}")

            # Statistical comparisons using paired t-tests
            # 1. Leaf vs Overall
            leaf_vs_overall = pg.ttest(leaf_data['Score'], [best_overall_cond] * len(leaf_data), paired=True)
            print(f"Leaf vs. Overall: T = {leaf_vs_overall['T'][0]}, P-Value = {leaf_vs_overall['p-val'][0]}")

            # 2. Leaf vs Sensor
            leaf_vs_sensor = pg.ttest(leaf_data['Score'], [best_sensor_cond] * len(leaf_data), paired=True)
            print(f"Leaf vs. Sensor: T = {leaf_vs_sensor['T'][0]}, P-Value = {leaf_vs_sensor['p-val'][0]}")

            # 3. Sensor vs Overall
            sensor_vs_overall = pg.ttest([best_sensor_cond] * len(leaf_data), [best_overall_cond] * len(leaf_data), paired=True)
            print(f"Sensor vs. Overall: T = {sensor_vs_overall['T'][0]}, P-Value = {sensor_vs_overall['p-val'][0]}")

            # Append data for plotting
            leaves.append(leaf)
            best_overall_scores.append(best_overall_cond)
            best_sensor_scores.append(best_sensor_cond)
            best_leaf_scores.append(best_leaf_score)
            best_overall_std.append(best_overall_cond_std)
            best_sensor_std.append(best_sensor_cond_std)
            best_leaf_std.append(leaf_data['Score'].std())

        # Plot the bar graph for best scores
        x = range(len(leaves))
        bar_width = 0.2

        bars1 = plt.bar(x, best_overall_scores, width=bar_width, label='Best Overall Conditions',
                        align='center', yerr=best_overall_std, capsize=5)
        bars2 = plt.bar([p + bar_width for p in x], best_sensor_scores, width=bar_width,
                        label=f'Best {sensor} Conditions', align='center', yerr=best_sensor_std, capsize=5)
        bars3 = plt.bar([p + 2 * bar_width for p in x], best_leaf_scores, width=bar_width,
                        label='Best Leaf Conditions', align='center', yerr=best_leaf_std, capsize=5)

        # Add labels and title
        plt.xticks([p + bar_width for p in x], leaves)
        plt.ylabel('Score')
        plt.title(f'Comparison of Best Conditions for {sensor} Across Leaves')
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.ylim([0.5, 1])
        plt.show()


def perform_anova_and_tukey_test(df: pd.DataFrame, factor: str, sensor: str = None) -> None:
    """
    Performs an ANOVA test on the specified factor for a given sensor (or all sensors if not specified),
    and if significant, runs Tukey's HSD test to compare different levels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    factor (str): The factor to compare between, such as "LED Current" or "Integration Time".
    sensor (str, optional): The specific sensor to analyze. If None, the analysis is performed across all sensors.

    Returns:
    None
    """
    # Filter data by sensor if specified
    if sensor:
        df = df[df['Sensor'] == sensor]
        if df.empty:
            print(f"No data available for the specified sensor: {sensor}")
            return
        print(f"\nRunning analysis for Sensor: {sensor}")
    else:
        print("\nRunning analysis for all sensors")

    # Perform the ANOVA test with detailed output
    anova_results = pg.anova(data=df, dv='Score', between=factor, detailed=True)
    p_value = anova_results['p-unc'][0]

    # Print the ANOVA results
    print(f"\nANOVA Test Results for Factor: {factor}")
    print(anova_results[['Source', 'SS', 'DF', 'MS', 'F', 'p-unc']])

    # Check if the ANOVA test is significant
    if p_value < 0.5:
        print(f"\nThe ANOVA test is significant (p = {p_value:.3f}). "
              f"Proceeding with Tukey's HSD test.")

        # Perform the Tukey's HSD test
        tukey_results = pg.pairwise_tukey(data=df, dv='Score', between=factor)

        # Print the Tukey's HSD results
        print(f"\nTukey's HSD Test Results for Factor: {factor}")
        print(tukey_results[['A', 'B', 'diff', 'p-tukey']])

        # Additional interpretation
        print("\nInterpretation:")
        for index, row in tukey_results.iterrows():
            if row['p-tukey'] < 0.05:
                print(f"{row['A']} performed significantly better than {row['B']} (p = {row['p-tukey']:.3f}).")
            else:
                print(f"No significant difference between {row['A']} and {row['B']} (p = {row['p-tukey']:.3f}).")
    else:
        print(f"\nThe ANOVA test is not significant (p = {p_value:.3f}). Tukey's HSD test will not be performed.")


if __name__ == '__main__':
    # make_anova_table_files()

    if False:
        # List of conditions to iterate over and generate plots for each
        conditions_to_plot = ['LED Current', 'Integration Time', 'Measurement Type', 'Preprocess',
                              'Cross Validation']

        for _condition in conditions_to_plot:
            # Collect the best scores for each condition
            _condition_best_scores = collect_best_scores_for_each_condition(condition)

            # Plot the performance for the current condition
            plot_performance_for_condition(_condition, _condition_best_scores)
    if False:
        with PdfPages("best_results.pdf") as pdf:
            for _leaf in ALL_LEAVES:
                for _sensor in SENSORS:
                    print(f"{_leaf}: {_sensor}")
                    _df = pd.read_csv(f"ANOVA_data/ANOVA_{_leaf}_{_sensor}.csv")

                    # plot_scores_for_leaf_sensor(df, leaf, sensor, pdf=pdf)
    # paiwise_ttests()
    # print_pg_anova_tables()
    # print_1_way_anova('Regression Model')
    best_conditions_dict = get_best_conditions()
    print("Best Overall Sensor Conditions:\n", best_conditions_dict['best_sensor_conditions'])
    print("\nBest Overall Conditions for All Leaves and Sensors:\n",
          best_conditions_dict['best_overall_conditions'])
    # print(best_sensor_conditions)
    # plot_and_check_statistics_for_conditions()
    _df = get_combined_anova_tables()
    _df = _df[df['Integration Time'].isin([50])]
    _df = _df[df['LED Current'].isin(["12.5 mA"])]
    print(_df['Regression Model'].unique())
    # df = df[df['Leaf'].isin(['mango'])]
    # df = df[df['Preprocess'].isin(["Poly"])]

    # check_statistical_significance(df)
    # find_best_conditions_for_led_and_integration_time()
    print(_df.columns)
    for _sensor in SENSORS:
        print(f"========= SENSOR {sensor} ============")
        perform_anova_and_tukey_test(_df, ['Regression Model'],
                                     sensor=_sensor)
