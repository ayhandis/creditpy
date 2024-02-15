import pandas as pd
import numpy as np

def PSI_calc_data(main_data, second_data, bins, default_flag):
    """
    Calculate the PSI (Population Stability Index) for each binned variable in the datasets.

    Parameters:
    - main_data (pandas.DataFrame): The main dataset.
    - second_data (pandas.DataFrame): The second dataset.
    - bins (dict or int): A dictionary containing the binning information for each variable,
                          or the number of bins to use for binning.
    - default_flag (str): The default flag variable to exclude from the calculation.

    Returns:
    pandas.DataFrame: A DataFrame containing the binned variables and their corresponding PSI values.
    """
    def calculate_psi(data1, data2, variable, bins):
        if isinstance(bins, int):
            y = pd.cut(data1[variable], bins=bins, include_lowest=True, right=True).value_counts(normalize=True)
            u = pd.cut(data2[variable], bins=bins, include_lowest=True, right=True).value_counts(normalize=True)
        else:
            y = pd.cut(data1[variable], bins=bins[variable], include_lowest=True, right=True).value_counts(normalize=True)
            u = pd.cut(data2[variable], bins=bins[variable], include_lowest=True, right=True).value_counts(normalize=True)
        merged = pd.merge(left=y, right=u, how='outer', left_index=True, right_index=True)
        merged['percenty'] = merged.iloc[:, 1].fillna(0)
        merged['percentx'] = merged.iloc[:, 0].fillna(0)
        merged['SSI'] = (merged['percentx'] - merged['percenty']) * np.log(merged['percentx'] / merged['percenty'])
        # Handle division by zero
        merged['SSI'] = np.where(merged['percenty'] == 0, 0, merged['SSI'])
        psi = abs(merged['SSI'].sum()) * 100
        return psi

    psi_values = {}
    for variable in main_data.columns:
        if variable != default_flag:
            psi_values[variable] = calculate_psi(main_data, second_data, variable, bins)

    psi_df = pd.DataFrame(list(psi_values.items()), columns=['Variable', 'PSI'])
    return psi_df
