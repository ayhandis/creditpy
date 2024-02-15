import pandas as pd
from scipy.stats import norm

def Binomial_test(rating_scale_data, total_observations, PD, DR, confidence_level=0.90, tail="one"):
    """
    Perform binomial test on master scale data.

    Parameters:
    - rating_scale_data (DataFrame): Master scale data set.
    - total_observations (str): Total observations variable.
    - PD (str): PD variable.
    - DR (str): Default rate variable.
    - confidence_level (float): Confidence level, default is 0.90.
    - tail (str): Tail preference of the test, 'one' or 'two', default is 'one'.

    Returns:
    DataFrame: DataFrame containing test results.
    """
    def one_tail_test(data, total_col, pd_col, dr_col, conf_level):
        data['BadObs'] = data[total_col] * data[dr_col]
        data['BadEstimations'] = data[total_col] * data[pd_col]
        data['TestEstimation'] = data[pd_col] * data[total_col] + norm.ppf(conf_level) * \
                                 (data[pd_col] * data[total_col] * (1 - data[pd_col])) ** 0.5

        dif_TestEstimation = data['TestEstimation'] - data['BadObs']
        data['Test_Result'] = pd.cut(dif_TestEstimation, bins=[-float('inf'), 0, float('inf')],
                                      labels=["Target Value Underestimated", "Target Value Correct"],
                                      include_lowest=True, right=False)
        return data

    def two_tail_test(data, total_col, pd_col, dr_col, conf_level):
        data['BadObs'] = data[total_col] * data[dr_col]
        data['BadEstimations'] = data[total_col] * data[pd_col]
        data['TestEstimationUpper'] = data[pd_col] * data[total_col] + norm.ppf(conf_level) * \
                                      (data[pd_col] * data[total_col] * (1 - data[pd_col])) ** 0.5
        data['TestEstimationLower'] = data[pd_col] * data[total_col] - norm.ppf(conf_level) * \
                                      (data[pd_col] * data[total_col] * (1 - data[pd_col])) ** 0.5
        data['Test_Result'] = pd.cut(data['BadObs'], bins=[-float('inf'), data['TestEstimationLower'],
                                                           data['TestEstimationUpper'], float('inf')],
                                      labels=["Target Value Overestimated", "Target Value Correct",
                                              "Target Value Underestimated"], include_lowest=True, right=False)
        return data

    if tail == "one":
        return one_tail_test(rating_scale_data, total_observations, PD, DR, confidence_level)
    elif tail == "two":
        return two_tail_test(rating_scale_data, total_observations, PD, DR, confidence_level)
    else:
        print("Tail can only be 'one' or 'two'.")



