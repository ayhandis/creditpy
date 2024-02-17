import pandas as pd
import numpy as np


def IV_elimination(data, default_flag, iv_threshold):
    """
    Eliminate variables with Information Value (IV) less than a given threshold.

    Parameters:
    - data (DataFrame): The dataset.
    - default_flag (str): The name of the default flag variable.
    - iv_threshold (float): The IV threshold for elimination.

    Returns:
    DataFrame: The dataset with variables eliminated based on IV threshold.
    """

    def IV_calc(data, default_flag, variable):
        good = data[data[default_flag] == "0"]
        bad = data[data[default_flag] == "1"]

        x = good[variable].value_counts().to_frame().reset_index()
        x.columns = ['Var1', 'Freq.x']
        y = bad[variable].value_counts().to_frame().reset_index()
        y.columns = ['Var1', 'Freq.y']

        merged = pd.merge(x, y, on="Var1", how="outer")
        merged['percentx'] = merged['Freq.x'] / merged['Freq.x'].sum()
        merged['percenty'] = merged['Freq.y'] / merged['Freq.y'].sum()
        merged['IV'] = (merged['percentx'] - merged['percenty']) * np.log(merged['percentx'] / merged['percenty'])
        IV_RESULT = merged['IV'].sum()

        return IV_RESULT

    iv_table = pd.DataFrame(columns=['Variable', 'IV'])

    for column in data.columns:
        if column != default_flag:
            iv = IV_calc(data, default_flag, column)
            iv_table = iv_table.append({'Variable': column, 'IV': iv}, ignore_index=True)

    iv_table = iv_table[iv_table['Variable'] != default_flag]

    elimination_list = iv_table[iv_table['IV'] < iv_threshold]['Variable']
    eliminated_data = data.drop(columns=elimination_list)

    return eliminated_data

