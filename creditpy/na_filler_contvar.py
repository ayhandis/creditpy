import numpy as np
from scipy.stats import ks_2samp

def na_filler_contvar(data, variable, pvalue=0.05):
    """
    Fill missing values of a continuous variable in a given dataset.

    This function fills missing values (NA) of a continuous variable in a dataset.
    It uses the Kolmogorov-Smirnov test to determine if the variable is distributed normally.
    If the variable is normally distributed, missing values are filled with the mean,
    otherwise, they are filled with the median.

    Parameters:
    data (DataFrame): The dataset containing the variable with missing values.
    variable (str): The name of the continuous variable to fill missing values for.
    pvalue (float, optional): The p-value threshold for the Kolmogorov-Smirnov test.
                               Default is 0.05.

    Returns:
    DataFrame: The dataset with missing values of the specified variable filled.

    Example:
    >>> import pandas as pd
    >>> example_data = pd.DataFrame({
    ...     'name': ['John Doe', 'Peter Gynn', 'Jolie Hope'],
    ...     'birth_year': [1980, 1985, 1971],
    ...     'salary': [20000, np.nan, 10000]
    ... })
    >>> na_filler_contvar(example_data, "salary")
          name  birth_year   salary
    0   John Doe        1980  20000.0
    1  Peter Gynn        1985  15000.0
    2  Jolie Hope       1971  10000.0
    """
    avg = np.nanmean(data[variable])
    median = np.nanmedian(data[variable])
    normal_dist = np.random.normal(size=data[variable].notna().sum())
    ks_statistic, ks_pvalue = ks_2samp(data[variable].dropna(), normal_dist)
    median_avg = median if ks_pvalue < pvalue else avg
    data[variable] = data[variable].fillna(median_avg)
    return data
