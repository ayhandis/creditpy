import pandas as pd
import numpy as np


def IV_calc(data, default_flag, variable):
    """
    Calculate the Information Value (IV) for a given variable.

    Parameters:
    - data (DataFrame): The dataset.
    - default_flag (str or numeric): The name of the default flag variable.
    - variable (str): The name of the variable for which IV is to be calculated.

    Returns:
    float: The calculated Information Value (IV).
    """
    good = data[data[default_flag] == 0]
    bad = data[data[default_flag] == 1]

    print("Good data:")
    print(good.head())
    print("\nBad data:")
    print(bad.head())

    x = good[variable].value_counts().to_frame().reset_index()
    x.columns = ['Var1', 'Freq.x']
    y = bad[variable].value_counts().to_frame().reset_index()
    y.columns = ['Var1', 'Freq.y']

    print("\nCounts for variable", variable, "in good data:")
    print(x)
    print("\nCounts for variable", variable, "in bad data:")
    print(y)

    merged = pd.merge(x, y, on="Var1", how="outer")
    merged['percentx'] = merged['Freq.x'] / merged['Freq.x'].sum()
    merged['percenty'] = merged['Freq.y'] / merged['Freq.y'].sum()

    print("\nMerged data:")
    print(merged)

    merged['IV'] = (merged['percentx'] - merged['percenty']) * np.log(merged['percentx'] / merged['percenty'])
    IV_RESULT = merged['IV'].sum()

    print("\nIV for variable", variable, ":", IV_RESULT)

    return IV_RESULT


def IV_calc_data(data, default_flag):
    """
    Calculate the Information Value (IV) for each variable in the dataset.

    Parameters:
    - data (DataFrame): The dataset.
    - default_flag (str or numeric): The name of the default flag variable.

    Returns:
    DataFrame: DataFrame containing variables and their corresponding IVs.
    """
    iv_column_names = []
    Information_Values = []

    for column in data.columns:
        if column != default_flag:
            iv = IV_calc(data, default_flag, column)
            iv_column_names.append(column)
            Information_Values.append(iv)

    iv_summary = pd.DataFrame({'Variable': iv_column_names, 'IV': Information_Values})
    iv_summary = iv_summary[iv_summary['Variable'] != default_flag].reset_index(drop=True)

    return iv_summary
