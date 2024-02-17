import pandas as pd
import numpy as np
from scipy.stats import chi2


def chisquare_test(data, PD, observed_bad, total_observations, confidence_level=0.95):
    """
    Chi Square Test

    This function allows performing a chi-square test on master scale data.

    Parameters:
    data : pandas DataFrame
        A master scale data set.
    PD : str
        Name of the PD variable.
    observed_bad : str
        Name of the observed defaults variable.
    total_observations : str
        Name of the total observations variable.
    confidence_level : float, optional
        Confidence level for the test, default is 0.95.

    Returns:
    dict
        A dictionary containing the following keys:
        - 'data': DataFrame with calculated fields
        - 'p_value': The p-value of the chi-square test
        - 'result': The result of the test

    Examples:
    chisquare_test(master_scale_data, "PD", "Bad_obs", "Tot_obs", 0.90)
    """
    data['expected_bad'] = data[total_observations] * data[PD]
    data['chi_square'] = (((data['expected_bad'] - data[observed_bad]) ** 2) / data['expected_bad'])
    chi2_statistic = np.sum(((data['expected_bad'] - data[observed_bad]) ** 2) / data['expected_bad'])
    p_value = 1 - chi2.cdf(chi2_statistic, df=1)  # Calculating p-value
    cl = 1 - confidence_level
    if p_value > cl:
        result = f"The rating scale did not pass the test {round(p_value, 3)} > {cl}"
    else:
        result = f"The rating scale passed the test. {round(p_value, 3)} < {cl}"

    return {
        'data': data,
        'p_value': p_value,
        'result': result
    }
