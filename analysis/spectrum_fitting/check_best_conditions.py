# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
Make an Anova table for AS7262, AS7263 and AS7265x sensor for different
led current, measurement types, and integration times for the different leave types
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
from collections import defaultdict
import itertools
import warnings

# installed libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ARDRegression, HuberRegressor, LassoLarsIC, LinearRegression
from sklearn.model_selection import (cross_val_score,
                                     GroupKFold, GroupShuffleSplit)
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator  # For cleaner annotations


# local files
import get_data
import remove_outliers
from stratified_group_shuffle_split import StratifiedGroupShuffleSplit


warnings.filterwarnings(action='ignore', category=DataConversionWarning)
LED_CURRENTS = ["12.5 mA", "25 mA", "50 mA", "100 mA"]
INT_TIMES = [50, 100, 150, 200, 250]
ALL_LEAVES = ["mango", "banana", "jasmine", "rice", "sugarcane"]
MEASUREMENT_TYPES = ["raw", "reflectance", "absorbance"]
SENSORS = ["as7262", "as7263", "as7265x"]
pls_n_comps = {"as7262": {"banana": 6, "jasmine": 4, "mango": 5, "rice": 6, "sugarcane": 6},
               "as7263": {"banana": 5, "jasmine": 5, "mango": 5, "rice": 5, "sugarcane": 5},
               "as7265x": {"banana": 8, "jasmine": 14, "mango": 11, "rice": 6, "sugarcane": 6}}

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

deep_palette = sns.color_palette("deep")
deep_blue = deep_palette[0]
deep_yellow = deep_palette[8]
deep_red = deep_palette[3]
violin_colors = {"as7262": [[deep_blue, deep_red, deep_red, deep_red],
                            [deep_blue, deep_blue, deep_blue, deep_blue, deep_blue],
                            [deep_blue, deep_blue, deep_blue]],
                 "as7263": [[deep_blue, deep_blue, deep_red, deep_red],
                            [deep_red, deep_red, deep_blue, deep_blue, deep_blue],
                            [deep_red, deep_red, deep_blue]],
                 "as7265x": [[deep_blue, deep_red, deep_red],
                             [deep_blue, deep_blue, deep_blue],
                             [deep_blue, deep_blue, deep_blue]]}
YLIM = ([0.5, 1.0])


def make_anova_table_files():
    """ make .csv file for each sensor to run ANOVA analysis on"""
    for sensor in ["as7263"]:
        make_anova_table_file(sensor)


def make_anova_table_file(sensor: str, cv_repeats: int = 10):
    """
    Generate an ANOVA table CSV file for a given sensor and leaf combinations
    by running Partial Least Squares (PLS) regression models with various
    parameter settings.

    Parameters
    ----------
    sensor : str
        The name of the sensor used for data collection (e.g., 'as7262', 'as7265x').
    cv_repeats : int, optional
        The number of cross-validation splits for StratifiedGroupShuffleSplit
        during model evaluation (default is 10).

    Outputs
    -------
    Saves a CSV file named "ANOVA_<sensor>.csv" in the "ANOVA_data" directory.
    The file includes columns:
        - Leaf: The leaf sample analyzed.
        - Measurement Type: The type of measurement (e.g., 'reflectance', 'absorbance', 'raw').
        - Integration Time: The integration time for the measurement.
        - LED Current: The current applied to the LED (e.g., '12.5 mA', '25 mA').
        - Score: The average cross-validation score for the given parameter combination.
    """
    results = []
    for leaf in ALL_LEAVES:
        print(f"Making {leaf} {sensor} ANOVA table")

        int_times = INT_TIMES
        led = "White LED"
        cv = StratifiedGroupShuffleSplit(n_splits=cv_repeats, test_size=0.2,
                                         n_bins=10)
        if sensor == "as7265x":
            int_times = [50, 100, 150]
            led = "b'White IR'"
        combinations = itertools.product(MEASUREMENT_TYPES, int_times, LED_CURRENTS)

        for measure_type, int_time, current in combinations:
            if sensor == "as7265x" and current == "100 mA":  # was not tested
                continue

            x, y, groups = get_data.get_x_y(
                leaf=leaf, sensor=sensor,
                led=led,
                measurement_type=measure_type,
                int_time=int_time, led_current=current,
                send_leaf_numbers=True)
            y = y["Avg Total Chlorophyll (µg/cm2)"]

            # Calculate residuals for each group
            residues = remove_outliers.calculate_residues(x, groups)

            # Apply Mahalanobis outlier removal on residuals
            mask = remove_outliers.mahalanobis_outlier_removal(residues)

            x = x[mask]  # Cleaned feature data
            y = y[mask]  # Cleaned target data
            groups = groups[mask]  # Cleaned group information

            # Further preprocessing based on the 'preprocess' parameter
            regressor = PLSRegression(n_components=pls_n_comps[sensor][leaf])

            # Scaling the data
            x_current = StandardScaler().fit_transform(x)

            # for i in range(repeats):

            # Predict using cross-validation
            scores = cross_val_score(
                regressor, x_current, y, groups=groups, cv=cv,
                scoring=SCORE)
            for score in scores:
                # Append the results
                results.append({
                    "Leaf": leaf,
                    "Measurement Type": measure_type,
                    "Integration Time": int_time,
                    "LED Current": current,
                    "Score": score
                })
            print(len(results))

    # Create a DataFrame from the results and save it to a CSV file
    results_df = pd.DataFrame(results)
    filename = f"ANOVA_data/ANOVA_{sensor}.csv"
    results_df.to_csv(filename, index=False)


def print_pg_anova_table(sensor: str):
    """
    Performs and prints in LaTex an ANOVA analysis for a given sensor type.

    This function reads ANOVA results from a file, applies statistical analysis
    (ANOVA and pairwise tests), formats the results, and saves the filtered
    results to a CSV file while also printing formatted tables.

    Parameters
    ----------
    sensor : str
        The sensor type for which the ANOVA analysis is performed.

    Returns
    -------
    None
        The function prints formatted ANOVA results and pairwise comparison tables
        (in LaTeX format) and saves filtered ANOVA results to a CSV file.

    Notes
    -----
    - The input CSV file is expected to be named "ANOVA_<sensor>.csv" and located
      in the "ANOVA_data" directory.
    - The p-values in the ANOVA results are adjusted for multiple comparisons using
      the Bonferroni correction method.
    - Pairwise comparisons are skipped for interacting factors (those containing "*").

    Workflow
    --------
    1. Read ANOVA results from "ANOVA_<sensor>.csv".
    2. Perform ANOVA using the dependent variable 'Score' and factors:
       - 'Leaf'
       - 'Measurement Type'
       - 'Integration Time'
       - 'LED Current'
    3. Apply Bonferroni correction to the p-values and add a new column, 'p-corrected'.
    4. Format the ANOVA table:
       - Convert 'DF' to integers.
       - Round 'SS' and 'F' values to two decimal places.
    5. Print the formatted ANOVA table.
    6. Extract and print non-interacting components (factors without "*") as a LaTeX table.
    7. Filter the ANOVA table to include only rows where 'p-corrected' < 0.05.
    8. For each significant non-interacting factor, perform pairwise comparisons
       and print results in LaTeX format.
    9. Save the filtered ANOVA table to "ANOVA_table_<sensor>.csv" in the "ANOVA_data" directory.
    """

    def conditional_format(value):
        """
        Format a value based on its magnitude:
        - Values > 0.01 are formatted as decimals with 3 decimal places.
        - Values <= 0.01 are formatted in scientific notation with 2 decimal places.
        - Values smaller than NumPy's floating-point tolerance are displayed as '< ε',
          and the tolerance is printed.

        Parameters
        ----------
        value : float
            The value to be formatted.

        Returns
        -------
        str
            The formatted string representation of the value.
        """
        tolerance = np.finfo(float).eps  # Numerical tolerance for float64

        if abs(value) < tolerance:
            print(f"Value {value:.2e} is smaller than the tolerance (ε = {tolerance:.2e})")
            return r"$< \varepsilon$"  # LaTeX string for '< ε'
        elif value > 0.01:
            return "{:.3f}".format(value)  # Decimal format with 3 decimal places
        else:
            return "{:.2e}".format(value)  # Scientific notation
    print(f"anova for {sensor}")
    filename = f"ANOVA_data/ANOVA_{sensor}.csv"
    df = pd.read_csv(filename)
    between_columns = ['Leaf', 'Measurement Type', 'Integration Time', 'LED Current']
    if False:  # filter the AS7263 data, its super confusing
        if sensor == "as7263":
            df = df[df['Measurement Type'] == "absorbance"]
            between_columns = ['Leaf', 'Integration Time', 'LED Current']

    # Perform the ANOVA test
    aov = pg.anova(dv='Score',
                   between=between_columns,
                   data=df)

    # Apply Bonferroni correction to the p-values
    aov['p-corrected'] = pg.multicomp(aov['p-unc'], method='bonferroni')[1]

    # Filter rows where p-corrected > 0.05
    filtered_aov = aov[aov['p-corrected'] < 0.05]
    # format columns for latex
    aov['DF'] = pd.to_numeric(aov['DF'], errors='coerce').astype(int)
    print(aov)
    for column in ['SS', 'F', 'MS']:
        aov[column] = aov[column].apply(lambda x: f"{x:.2f}")  # 2 decimal places for SS

    print(aov)

    # Filter non-interacting components (rows without "*")
    non_interacting = aov[~aov['Source'].str.contains("\*")]
    for column in ["p-unc", "p-corrected"]:
        non_interacting[column] = non_interacting[column].apply(conditional_format)

    # Print LaTeX table
    latex_table = non_interacting.to_latex(index=False, float_format="{:.2e}".format,
                                           escape=False,
                                           caption="ANOVA results for non-interacting components.",
                                           label=f"tab:{sensor}_anova")
    print(latex_table)
    print(filtered_aov)

    # Run pairwise tests for significant variables
    pairwise_results = {}
    for factor in filtered_aov['Source']:
        if '*' in factor:
            continue
        print(f"\nRunning pairwise comparisons for: {factor}")
        posthoc = pg.pairwise_tests(data=df, dv='Score', between=factor, padjust='bonf')
        pairwise_results[factor] = posthoc

        print(posthoc[["Contrast", "A", "B", "T", "p-unc", "p-corr", "BF10"]])
        # for column in ['p-unc', 'p-corr']:
        #     # Scientific notation
        #     posthoc[column] = posthoc[column].map(
        #         lambda x: "1.0" if x == 1 else f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "$0$")
        # Format specific columns
        posthoc["T"] = posthoc["T"].map(
            "{:.2f}".format)  # Format 'T' with 2 decimal places
        for column in ["p-unc", "p-corr"]:
            posthoc[column] = posthoc[column].apply(conditional_format)
        pw_latex = posthoc[["Contrast", "A", "B", "T", "p-unc", "p-corr", "BF10"]].to_latex(
            index=False, float_format="{:.2e}".format, escape=False,
            caption=f"{sensor} {factor} Pairwise tests.",
            label=f"tab:{sensor}_{factor}_pairwise_tests")
        print(pw_latex)

    # Save the filtered ANOVA table to a new CSV file
    output_filename = f"ANOVA_data/ANOVA_table_{sensor}.csv"
    filtered_aov.to_csv(output_filename, index=False)


def make_box_plots_with_stats() -> None:
    """
    Generate violin plots for model performance scores by varying 'LED Current',
    'Integration Time', and 'Measurement Type' for each sensor in a 3x3 grid.
    Add statistical significance markers using pairwise tests.

    Returns
    -------
    None
        Displays a 3x3 figure of violin plots with statistical annotations.
    """
    sensors = ["as7262", "as7263", "as7265x"]  # Sensor names
    best_led_current = "12.5 mA"
    best_int_time = 50
    best_measure_type = "absorbance"

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 9),
                             sharey=True)
    fig.suptitle("Statistical Analysis of Best Conditions", fontsize=12, y=0.95)

    def plot_boxplots_with_stats(df: pd.DataFrame, axes_row: list[plt.Axes], sensor: str) -> None:
        """
        Plot violin plots for 'LED Current', 'Integration Time', and 'Measurement Type'
        with statistical annotations.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing relevant columns.
        axes_row : list[plt.Axes]
            A list of three matplotlib Axes objects for the current sensor row.
        sensor : str
            The name of the sensor being processed.
        """
        print(df)
        # Vary 'LED Current'
        df_led = df[(df['Integration Time'] == best_int_time) &
                    (df['Measurement Type'] == best_measure_type)]
        # pairs_led = [(a, b) for a in df_led['LED Current'].unique()
        # for b in df_led['LED Current'].unique() if a < b]
        sns.violinplot(data=df_led, x='LED Current', y='Score', ax=axes_row[0],
                       palette=violin_colors[sensor][0], legend=False)
        # add_stat_annotations(df_led, "LED Current", pairs_led, axes_row[0])
        # axes_row[0].set_title(f"{sensor.upper()} - LED Current")
        axes_row[0].set_ylabel("Score")
        # Group by the column and calculate the mean score
        grouped = df_led.groupby('LED Current')['Score'].mean()
        print(grouped)
        # best_condition = grouped.idxmax()  # Get the group with the highest mean score
        # Print the best condition
        # print(f"Best {'LED Current'}: {best_condition} (Mean Score: {grouped.max():.4f})")

        # Vary 'Integration Time'
        df_time = df[(df['LED Current'] == best_led_current) &
                     (df['Measurement Type'] == best_measure_type)]
        # convert integration cyles to milliseconds
        df_time['Integration Time'] = (2.8 * df['Integration Time']).astype(int).astype(str) + " ms"
        pairs_time = [(a, b) for a in df_time['Integration Time'].unique()
                      for b in df_time['Integration Time'].unique() if a < b]

        sns.violinplot(data=df_time, x='Integration Time', y='Score', ax=axes_row[1],
                       palette=violin_colors[sensor][1], legend=False)
        # add_stat_annotations(df_time, "Integration Time", pairs_time, axes_row[1])
        # axes_row[1].set_title(f"{sensor.upper()} - Integration Time")
        # grouped = df_led.groupby('Integration Time')['Score'].mean()
        # best_condition = grouped.idxmax()  # Get the group with the highest mean score

        # Print the best condition
        # print(f"Best {'Integration Time'}: {best_condition} (Mean Score: {grouped.max():.4f})")

        # Vary 'Measurement Type'
        df_type = df[(df['LED Current'] == best_led_current) &
                     (df['Integration Time'] == best_int_time)]
        # pairs_type = [(a, b) for a in df_type['Measurement Type'].unique()
        #               for b in df_type['Measurement Type'].unique() if a < b]
        sns.violinplot(data=df_type, x='Measurement Type', y='Score', ax=axes_row[2],
                       palette=violin_colors[sensor][2], legend=False)
        # add_stat_annotations(df_type, "Measurement Type", pairs_type, axes_row[2])
        # axes_row[2].set_title(f"{sensor.upper()} - Measurement Type")
        # grouped = df_led.groupby('Measurement Type')['Score'].mean()
        # best_condition = grouped.idxmax()  # Get the group with the highest mean score

        # Print the best condition
        # print(f"Best {'Measurement Type'}: {best_condition} (Mean Score: {grouped.max():.4f})")

        for ax in axes_row:
            ax.set_ylim(YLIM)

    def add_stat_annotations(df: pd.DataFrame, group_col: str,
                             pairs: list[tuple], ax: plt.Axes) -> None:
        """
        Add statistical significance annotations to a plot.

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the group and score columns.
        group_col : str
            Column name to group data for pairwise comparisons.
        pairs : list[tuple]
            List of pairs to compare.
        ax : plt.Axes
            Matplotlib Axes object to annotate.
        """
        # Run pairwise tests
        stats = pg.pairwise_tests(data=df, dv="Score", between=group_col, padjust="bonferroni")
        stats = stats[stats['p-unc'] < 0.05]  # Filter for significance
        print(stats)
        # Align pairs and p-values
        valid_pairs = []
        pvalues = []

        for pair in pairs:
            # Check if the pair exists in the stats table
            stat_row = stats[
                ((stats['A'] == pair[0]) & (stats['B'] == pair[1])) |
                ((stats['A'] == pair[1]) & (stats['B'] == pair[0]))
                ]
            if not stat_row.empty:
                valid_pairs.append(pair)
                pvalues.append(stat_row['p-unc'].values[0])

        # Annotate using statannotations
        if valid_pairs:
            annotator = Annotator(ax, valid_pairs, data=df, x=group_col, y="Score")
            annotator.configure(test=None, text_format="simple", loc="outside", verbose=2)
            annotator.set_pvalues_and_annotate(pvalues)

    # Loop through each sensor
    for i, sensor in enumerate(sensors):
        df = pd.read_csv(f"ANOVA_data/ANOVA_{sensor}.csv")
        # df['Integration Time'] = 2.8*df['Integration Time'].astype(int)
        # call the inner function to make the plots
        plot_boxplots_with_stats(df, axes[i], sensor)
        # add sensor label
        axes[i][0].annotate(f"AS{sensor[2:]}", (0.03, 0.05),
                            fontsize=12, xycoords='axes fraction')

    for i in range(3):  # each column
        axes[i][0].set_xticklabels(axes[i][0].get_xticklabels(), rotation=20, ha="center")
        axes[i][1].set_xticklabels(axes[i][1].get_xticklabels(), rotation=30, ha="center")
        axes[i][2].set_xticklabels(axes[i][2].get_xticklabels(), rotation=15, ha="center")

        for j in range(3):  # remove labels
            axes[j][i].set_xlabel("")
            # axes[i][j].set_xticklabels(axes[i][j].get_xticklabels(), rotation=30, ha="center")
            axes[i][j].annotate(f"{chr(j*3+i+97)})", (0.03, 1.0),
                                fontsize=12, fontweight='bold',
                                xycoords='axes fraction')
            for side in ['top', 'right']:
                axes[i][j].spines[side].set_visible(False)
    axes[0][0].set_title("LED Current")
    axes[0][1].set_title("Integration Time")
    axes[0][2].set_title("Data Type")

    # add legend
    # Manually create legend handles
    same_patch = mpatches.Patch(color=deep_blue, label='Similar (p > 0.05)')
    # mid_patch = mpatches.Patch(color=deep_yellow, label='Different (p < 0.05)')
    different_patch = mpatches.Patch(color=deep_red, label='Different (p < 0.01)')
    # Add the legend to the plot
    axes[2][2].legend(handles=[same_patch, different_patch], title="Statistically:",
                      title_fontsize='small',
                      bbox_to_anchor=(1.05, -0.05), loc="lower right",
                      fontsize='small', frameon=False)

    # axes[2][1].set_xlabel('Integration Time')
    # axes[2][1].set_xlabel('Data Type')
    # axes[2][0].set_xlabel('LED Current (mA)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.subplots_adjust(left=0.1, bottom=0.07, right=0.97, top=0.85,
    #                     wspace=0.1, hspace=.2)

    fig.savefig("ANOVA_best_conditions.jpg", dpi=600)
    plt.show()


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


def get_combined_anova_tables():
    combined_data = []
    # make combined dataset
    for sensor in SENSORS:
        df = pd.read_csv(f"ANOVA_data/ANOVA_{sensor}.csv")
        df['Sensor'] = sensor
        combined_data.append(df)
    combined_df = pd.concat(combined_data, ignore_index=True)
    # Convert relevant columns to float32
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[numeric_cols] = combined_df[numeric_cols].astype(np.float32)
    return combined_df


def get_best_conditions() -> dict:
    """
    Combines the functionality to find:
    1. The best conditions for each sensor across all leaves (the best overall sensor conditions).
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
    print("\nBest Overall Conditions for All Leaves and Sensors:\n",
           best_conditions_dict['best_overall_conditions'])
    """
    # Get the DataFrame from the combined ANOVA tables
    df = get_combined_anova_tables()

    # --- Best Conditions for Each Sensor (Overall across all leaves) ---
    # Group by 'Sensor' and other conditions, calculate the average 'Score' across all leaves
    mean_scores_sensor = df.groupby(
        ['Sensor', 'LED Current', 'Integration Time',
         'Measurement Type']
    )['Score'].mean().reset_index()

    # Find the row with the highest average score for each sensor
    best_avg_conditions_for_sensor = mean_scores_sensor.loc[
        mean_scores_sensor.groupby('Sensor')['Score'].idxmax()]

    # --- Best Overall Conditions for All Leaves and Sensors ---
    # Group by all condition columns without 'Sensor' and 'Leaf',
    # then calculate the mean score across all leaves and sensors
    mean_scores_overall = df.groupby(
        ['LED Current', 'Integration Time',
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

    for sensor in SENSORS:
        print(sensor)
        # Load the data for the current leaf/sensor combination
        df = pd.read_csv(f"ANOVA_data/ANOVA_{sensor}.csv")
        print(df)
        print('===')
        print(df.groupby([condition_name]))
        print('++++')
        # Group by the condition and regression model,
        # then find the max score for each combination
        grouped = df.groupby([condition_name]
                             ).agg({'Score': 'max'}).reset_index()
        print(grouped)
        # Store the best score for each regression model under each condition
        for index, row in grouped.iterrows():
            condition_value = row[condition_name]
            best_score = row['Score']
            condition_best_scores[condition_value] = best_score

    return condition_best_scores


def plot_performance_for_condition(condition_name, condition_best_scores):
    # Prepare the plot for each condition
    plt.figure(figsize=(10, 6))
    print(condition_name)
    print(condition_best_scores)
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
        print('models: ', models)
        # Plotting the mean and std of best scores for each r
        # egression model under the current condition
        plt.errorbar(models, mean_scores, yerr=std_scores, fmt='o', capsize=5,
                     linewidth=2, markeredgewidth=2,
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
    Plots bar graphs comparing the best overall conditions, the best sensor conditions,
    and the maximum score for each leaf. Additionally, performs statistical
    significance tests (paired t-test) for three comparisons:
    - Best leaf scores vs. best overall condition scores
    - Best leaf scores vs. best sensor condition scores
    - Best sensor scores vs. best overall condition scores

    The function will:
    - Plot bar graphs for each sensor showing the scores for the best overall condition,
      the best sensor-specific condition, and the maximum score for each leaf.
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
        return (leaf_data['LED Current'] == condition['LED Current']) & \
               (leaf_data['Integration Time'] == condition['Integration Time']) & \
               (leaf_data['Measurement Type'] == condition['Measurement Type'])

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
        sensor_condition = best_sensor_conditions[
            best_sensor_conditions['Sensor'] == sensor].iloc[0]
        print("Best Sensor Condition:")
        print(sensor_condition)

        # Get scores for each leaf
        for leaf in df['Leaf'].unique():
            # Filter for the current leaf and sensor combination
            leaf_data = df[(df['Leaf'] == leaf) & (df['Sensor'] == sensor)]

            # --- Best Overall Conditions ---
            best_overall_cond = leaf_data[filter_data(
                leaf_data, best_overall_conditions)]['Score'].mean()
            best_overall_cond_std = leaf_data[filter_data(
                leaf_data, best_overall_conditions)]['Score'].std()

            # --- Best Sensor Conditions ---
            best_sensor_cond = leaf_data[filter_data(
                leaf_data, sensor_condition)]['Score'].mean()
            best_sensor_cond_std = leaf_data[filter_data(
                leaf_data, sensor_condition)]['Score'].std()

            # --- Best Leaf Condition (Max Score for the Leaf) ---
            best_leaf_cond = leaf_data.loc[leaf_data['Score'].idxmax()]
            best_leaf_cond_score = best_leaf_cond['Score']

            # Print the detailed conditions
            print(f"\nLeaf: {leaf}")
            print(f"Best Leaf Condition:")
            # This prints the entire row including all relevant condition parameters
            print(best_leaf_cond.tolist())

            # Perform paired t-tests for statistical significance
            comparison_data = pd.DataFrame({
                'Best_Leaf_Score': leaf_data['Score'],
                'Best_Overall_Score': [best_overall_cond] * len(leaf_data),
                'Best_Sensor_Score': [best_sensor_cond] * len(leaf_data)
            })

            # T-test: Best leaf vs. best overall
            ttest_leaf_vs_overall = pg.ttest(comparison_data['Best_Leaf_Score'],
                                             comparison_data['Best_Overall_Score'], paired=True)
            t_statistic_leaf_vs_overall = ttest_leaf_vs_overall['T'].iloc[0]
            p_value_leaf_vs_overall = ttest_leaf_vs_overall['p-val'].iloc[0]

            # T-test: Best leaf vs. best sensor
            ttest_leaf_vs_sensor = pg.ttest(comparison_data['Best_Leaf_Score'],
                                            comparison_data['Best_Sensor_Score'], paired=True)
            t_statistic_leaf_vs_sensor = ttest_leaf_vs_sensor['T'].iloc[0]
            p_value_leaf_vs_sensor = ttest_leaf_vs_sensor['p-val'].iloc[0]

            # T-test: Best sensor vs. best overall
            ttest_sensor_vs_overall = pg.ttest(comparison_data['Best_Sensor_Score'],
                                               comparison_data['Best_Overall_Score'], paired=True)
            t_statistic_sensor_vs_overall = ttest_sensor_vs_overall['T'].iloc[0]
            p_value_sensor_vs_overall = ttest_sensor_vs_overall['p-val'].iloc[0]

            # Print the statistical test results
            print(f"Statistical Tests for Leaf: {leaf}")
            print(f"Leaf vs. Overall - T-Statistic: {t_statistic_leaf_vs_overall}, "
                  f"P-Value: {p_value_leaf_vs_overall}")
            print(f"Leaf vs. Sensor - T-Statistic: {t_statistic_leaf_vs_sensor}, "
                  f"P-Value: {p_value_leaf_vs_sensor}")
            print(f"Sensor vs. Overall - T-Statistic: {t_statistic_sensor_vs_overall}, "
                  f"P-Value: {p_value_sensor_vs_overall}")

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
                        label=f'Best {sensor} Conditions', align='center',
                        yerr=best_leaf_std, capsize=5)
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
    Checks the statistical significance between the best leaf scores and the
    best overall condition scores using Pingouin's t-test for each sensor.

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
                f"Leaf: {leaf}, Best Leaf Score: {best_leaf_score}, "
                f"Best Overall Condition Score: {best_overall_score}")
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
        sensor_condition = best_sensor_conditions[
            best_sensor_conditions['Sensor'] == sensor].iloc[0]

        # Loop through each leaf
        for leaf in df['Leaf'].unique():
            # Filter data for the current leaf and sensor
            leaf_data = df[(df['Leaf'] == leaf) & (df['Sensor'] == sensor)]

            # Group by LED Current and Integration Time and calculate the mean score
            grouped = leaf_data.groupby(['LED Current', 'Integration Time']
                                        )['Score'].mean().reset_index()

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
            print(f"Best Leaf Condition: LED Current = {best_leaf_led}, "
                  f"Integration Time = {best_leaf_integration}, Score = {best_leaf_score}")
            print(f"Best Overall Score: {best_overall_cond}")
            print(f"Best Sensor Score: {best_sensor_cond}")

            # Statistical comparisons using paired t-tests
            # 1. Leaf vs Overall
            leaf_vs_overall = pg.ttest(leaf_data['Score'],
                                       [best_overall_cond] * len(leaf_data), paired=True)
            print(f"Leaf vs. Overall: T = {leaf_vs_overall['T'][0]}, "
                  f"P-Value = {leaf_vs_overall['p-val'][0]}")

            # 2. Leaf vs Sensor
            leaf_vs_sensor = pg.ttest(leaf_data['Score'],
                                      [best_sensor_cond] * len(leaf_data), paired=True)
            print(f"Leaf vs. Sensor: T = {leaf_vs_sensor['T'][0]}, "
                  f"P-Value = {leaf_vs_sensor['p-val'][0]}")

            # 3. Sensor vs Overall
            sensor_vs_overall = pg.ttest([best_sensor_cond] * len(leaf_data),
                                         [best_overall_cond] * len(leaf_data), paired=True)
            print(f"Sensor vs. Overall: T = {sensor_vs_overall['T'][0]}, "
                  f"P-Value = {sensor_vs_overall['p-val'][0]}")

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
                        label=f'Best {sensor} Conditions', align='center',
                        yerr=best_sensor_std, capsize=5)
        bars3 = plt.bar([p + 2 * bar_width for p in x], best_leaf_scores, width=bar_width,
                        label='Best Leaf Conditions', align='center',
                        yerr=best_leaf_std, capsize=5)

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
    Performs an ANOVA test on the specified factor for a given sensor
    (or all sensors if not specified),
    and if significant, runs Tukey's HSD test to compare different levels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    factor (str): The factor to compare between, such as "LED Current" or "Integration Time".
    sensor (str, optional): The specific sensor to analyze.
    If None, the analysis is performed across all sensors.

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
                print(f"{row['A']} performed significantly better than {row['B']} "
                      f"(p = {row['p-tukey']:.3f}).")
            else:
                print(f"No significant difference between {row['A']} and {row['B']} "
                      f"(p = {row['p-tukey']:.3f}).")
    else:
        print(f"\nThe ANOVA test is not significant (p = {p_value:.3f}). "
              f"Tukey's HSD test will not be performed.")


if __name__ == '__main__':
    # make_anova_table_files()
    print_pg_anova_table('as7265x')
    # make_box_plots_with_stats()

    if False:
        # List of conditions to iterate over and generate plots for each
        conditions_to_plot = ['LED Current', 'Integration Time', 'Measurement Type']

        for _condition in conditions_to_plot:
            # Collect the best scores for each condition
            _condition_best_scores = collect_best_scores_for_each_condition(_condition)

            # Plot the performance for the current condition
            plot_performance_for_condition(_condition, _condition_best_scores)
    if False:
        with PdfPages("best_results.pdf") as pdf:
            for _leaf in ALL_LEAVES:
                for _sensor in SENSORS:
                    print(f"{_leaf}: {_sensor}")
                    _df = pd.read_csv(f"trash/ANOVA_data/ANOVA_{_leaf}_{_sensor}.csv")

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
