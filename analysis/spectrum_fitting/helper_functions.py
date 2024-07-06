# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"

import pandas as pd

def filter_df(selected_row, df2):
    """
    Filter rows in df2 based on matching columns with selected_row from df1, excluding 'Scores'.

    Args:
        selected_row (pd.Series): Selected row from df1.
        df2 (pd.DataFrame): DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame containing rows from df2 that match selected_row.

    Example:
        # Example usage:
        filtered_df2 = filter_df(df1.iloc[0], df2)
        print(filtered_df2)

    Input:
        selected_row (pd.Series): A single row (as pd.Series) from df1.
        df2 (pd.DataFrame): DataFrame from which rows will be filtered based on selected_row.

    Output:
        pd.DataFrame: Filtered DataFrame containing rows from df2 that match selected_row.
    """
    # Drop 'Scores' column from selected_row
    selected_row_filtered = selected_row.drop('Scores')

    # Drop 'Scores' column from df2
    df2_filtered = df2.drop(columns=['Scores'])

    # Merge based on remaining columns
    merged_df = pd.merge(selected_row_filtered.to_frame().T, df2_filtered, on=list(selected_row_filtered.index), suffixes=('_selected', '_df2'))

    # Filter rows where all columns match except 'Scores'
    filter_condition = merged_df.apply(lambda row: all(row.iloc[0] == row.iloc[1]), axis=1)
    filtered_df2 = df2[filter_condition]

    return filtered_df2

# Example usage if this file is executed directly
if __name__ == "__main__":
    # Example DataFrames
    df1 = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Scores': [80, 75, 85, 90]
    })

    df2 = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Scores': [85, 70, 80, 95]
    })

    # Select a row from the first DataFrame
    selected_row = df1.iloc[0]

    # Filter df2 based on selected_row
    filtered_df2 = filter_df(selected_row, df2)

    print("Selected row from df1:")
    print(selected_row)
    print("\nFiltered rows in df2 (Expected Output):")
    print(filtered_df2)