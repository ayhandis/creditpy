import pandas as pd
import numpy as np

def scaled_score(data, PD, ceiling_score, increase):
    """
    Create scaled scores for a given dataset.

    This function calculates scaled scores based on the provided probability of default (PD) variable
    using the provided ceiling score and increase level.

    Parameters:
    data (DataFrame): The dataset containing the PD variable.
    PD (str): The name of the probability of default (PD) variable in the dataset.
    ceiling_score (float): The ceiling score for transformation.
    increase (float): The increase level for transformation.

    Returns:
    DataFrame: The dataset with the scaled score column added.

    Example:
    >>> import pandas as pd
    >>> # Assume data is defined
    >>> default_f = ['1', '0', '0', '1', '1', '0', '0', '1', '1']
    >>> birth_year = [1980, 1985, 1971, 1971, 1985, 1971, 1980, 1980, 1985]
    >>> PD = [0.1, 0.12, 0.2, 0.23, 0.28, 0.33, 0.39, 0.45, 0.54]
    >>> example_data = pd.DataFrame({'default_f': default_f, 'birth_year': birth_year, 'PD': PD})
    >>> scaled_score_data = scaled_score(example_data, "PD", 1000, 15)
    >>> print(scaled_score_data)
    """

    odds = (1 - data[PD]) / data[PD]
    factor = increase / np.log(2)
    offset_val = ceiling_score - (factor * np.log(increase))
    data['scaled_score'] = offset_val + factor * np.log(odds)
    return data

