import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def time_series_gini_roc(data, default_flag, PD, time):
    """
    Calculate the Gini coefficient by time from estimated values calculated by logistic regression using ROC AUC.

    This function calculates the Gini coefficient over time for estimated values obtained
    from logistic regression, based on the provided dataset, default flag, PD variable, and time variable,
    using ROC AUC.

    Parameters:
    data (DataFrame): The dataset containing the variables.
    default_flag (str): The name of the default flag variable in the dataset.
    PD (str): The name of the PD variable in the dataset.
    time (str): The name of the time variable in the dataset.

    Returns:
    DataFrame: A DataFrame containing Gini coefficients by time.
    #
    # Example:
    # >>> import pandas as pd
    # >>> # Assume data is defined
    # >>> default_f = ['1', '0', '0', '1', '1', '0', '0', '1', '1']
    # >>> birth_year = [1980, 1985, 1971, 1971, 1985, 1971, 1980, 1980, 1985]
    # >>> job = [1, 1, 2, 2, 2, 3, 3, 2, 3]
    # >>> pd = [0.5, 0.2, 0.4, 0.5, 0.7, 0.9, 0.2, 0.3, 0.3]
    # >>> example_data = pd.DataFrame({'default_f': default_f, 'birth_year': birth_year, 'job': job, 'pd': pd})
    # >>> gini_result = time_series_gini_roc(example_data, "default_f", "pd", "birth_year")
    # >>> print(gini_result)
    # """

    times = []
    time_gini = []

    for t in data[time].unique():
        time_data = data[data[time] == t]
        times.append(t)
        time_gini.append(2 * roc_auc_score(time_data[default_flag], time_data[PD]) - 1)

    time_return = pd.DataFrame({"time": times, "Gini": time_gini})
    average_gini = time_return["Gini"].mean()
    time_return = time_return.append({"time": "Average", "Gini": average_gini}, ignore_index=True)

    return time_return
