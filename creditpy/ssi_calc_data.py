import pandas as pd
import numpy as np

def SSI_calc_data(main_data, second_data, default_flag):
    """
    Calculate the SSI for each variable in the datasets.

    Parameters:
    - main_data (pandas.DataFrame): The main dataset.
    - second_data (pandas.DataFrame): The second dataset.
    - default_flag (str): The default flag variable to exclude from the calculation.

    Returns:
    pandas.DataFrame: A DataFrame containing the variables and their corresponding SSIs.
    """
    def calculate_ssi(data1, data2, variable):
        y = data1[variable].value_counts(normalize=True)
        u = data2[variable].value_counts(normalize=True)
        merged = pd.merge(left=y, right=u, how='outer', left_index=True, right_index=True)
        merged['percenty'] = merged.iloc[:, 1].fillna(0)
        merged['percentx'] = merged.iloc[:, 0].fillna(0)
        merged['SSI'] = (merged['percentx'] - merged['percenty']) * np.log(merged['percentx'] / merged['percenty'])
        return merged['SSI'].sum()

    ssi_values = {}
    for variable in main_data.columns:
        if variable != default_flag:
            ssi_values[variable] = calculate_ssi(main_data, second_data, variable)

    ssi_df = pd.DataFrame(list(ssi_values.items()), columns=['Variable', 'SSI'])
    return ssi_df

# Example usage:
# Assuming main_data and second_data are your datasets, and default_flag is the column to exclude
# ssi_result = SSI_calc_data(main_data, second_data, default_flag)
# print(ssi_result)
