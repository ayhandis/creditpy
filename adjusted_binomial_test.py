import pandas as pd
from scipy.stats import norm
from math import sqrt

def Adjusted_Binomial_test(rating_scale_data, total_observations, PD, DR, confidence_level=0.90, tail="one", r=0.40):
    """
    Perform adjusted binomial test on master scale data.

    Parameters:
    - rating_scale_data (DataFrame): Master scale data.
    - total_observations (str): Column name for total observations.
    - PD (str): Column name for PD variable.
    - DR (str): Column name for Default Rate variable.
    - confidence_level (float): Confidence level, default is 0.90.
    - tail (str): Tail preference of the test, 'one' or 'two', default is 'one'.
    - r (float): Default correlation between rating grades, default is 0.40.

    Returns:
    DataFrame: DataFrame with adjusted test results.
    """

    def one_tail_test(data, total_col, pd_col, dr_col, conf_level, r):
        data['BadObs'] = data[total_col] * data[dr_col]
        data['BadEstimations'] = data[total_col] * data[pd_col]
        data['Default_Correlation'] = r

        t = norm.ppf(data[pd_col])
        Q_param = norm.cdf((sqrt(r) * norm.ppf(conf_level) + t) / sqrt(1 - r))
        Adjusted_result = Q_param + 1 / (2 * data[total_col]) * (
                2 * Q_param - 1 + (Q_param * (1 - Q_param)) /
                norm.pdf((sqrt(r) * norm.ppf(1 - conf_level) - t) / sqrt(1 - r)) *
                ((2 * r - 1) * norm.ppf(1 - conf_level) - t * sqrt(r)) / sqrt(r * (1 - r)))

        data['TestEstimation'] = round(Adjusted_result * data[total_col], 2)

        dif_TestEstimation = data['TestEstimation'] - data['BadObs']
        data['Test_Result'] = pd.cut(dif_TestEstimation,
                                     bins=[-float('inf'), 0, float('inf')],
                                     labels=["Target Value Underestimated", "Target Value Correct"],
                                     include_lowest=True, right=False)
        return data

    def two_tail_test(data, total_col, pd_col, dr_col, conf_level, r):
        data['BadObs'] = data[total_col] * data[dr_col]
        data['BadEstimations'] = data[total_col] * data[pd_col]
        data['Default_Correlation'] = r

        t = norm.ppf(data[pd_col])
        Q_param = norm.cdf((sqrt(r) * norm.ppf(conf_level) + t) / sqrt(1 - r))
        Adjusted_resultUpper = Q_param + 1 / (2 * data[total_col]) * (
                2 * Q_param - 1 + (Q_param * (1 - Q_param)) /
                norm.pdf((sqrt(r) * norm.ppf(1 - conf_level) - t) / sqrt(1 - r)) *
                ((2 * r - 1) * norm.ppf(1 - conf_level) - t * sqrt(r)) / sqrt(r * (1 - r)))
        Adjusted_resultLower = Q_param + 1 / (2 * data[total_col]) * (
                2 * Q_param - 1 - (Q_param * (1 - Q_param)) /
                norm.pdf((sqrt(r) * norm.ppf(1 - conf_level) - t) / sqrt(1 - r)) *
                ((2 * r - 1) * norm.ppf(1 - conf_level) - t * sqrt(r)) / sqrt(r * (1 - r)))

        data['TestEstimationUpper'] = round(Adjusted_resultUpper * data[total_col], 2)
        data['TestEstimationLower'] = round(Adjusted_resultLower * data[total_col], 2)

        data['Test_Result'] = pd.cut(data['BadObs'],
                                     bins=[-float('inf'), data['TestEstimationLower'],
                                           data['TestEstimationUpper'], float('inf')],
                                     labels=["Target Value Overestimated", "Target Value Correct",
                                             "Target Value Underestimated"],
                                     include_lowest=True, right=False)
        return data

    if tail == "one":
        return one_tail_test(rating_scale_data, total_observations, PD, DR, confidence_level, r)
    elif tail == "two":
        return two_tail_test(rating_scale_data, total_observations, PD, DR, confidence_level, r)
    else:
        print("Tail can only be 'one' or 'two'.")
