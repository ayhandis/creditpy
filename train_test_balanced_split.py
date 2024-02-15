import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_test_balanced_split(data, default_flag, balance_count, seed_value=1, ratio=0.67):
    """
    Split the dataset into balanced train and test sets.

    This function separates the dataset into a balanced train and test set based on the provided default flag variable.

    Parameters:
    data (DataFrame): The dataset containing the variables.
    default_flag (str): The name of the default flag variable in the dataset.
    balance_count (int): Desired number of good or bad observations.
    seed_value (int): Seed value for replicability (default is 1).
    ratio (float): The percentage of data to be used for the train set (default is 0.67).

    Returns:
    dict: A dictionary containing the balanced train and test sets.

    Example:
    >>> import pandas as pd
    >>> # Assume data is defined
    >>> default_f = ['1', '0', '0', '1', '1', '0', '0', '1', '1']
    >>> birth_year = [1980, 1985, 1971, 1971, 1985, 1971, 1980, 1980, 1985]
    >>> PD = [0.1, 0.12, 0.2, 0.23, 0.28, 0.33, 0.39, 0.45, 0.54]
    >>> example_data = pd.DataFrame({'default_f': default_f, 'birth_year': birth_year, 'PD': PD})
    >>> train_test_sets = train_test_balanced_split(example_data, "default_f", balance_count=2, seed_value=1, ratio=0.90)
    >>> print(train_test_sets['train_balanced'])
    >>> print(train_test_sets['test'])
    """

    np.random.seed(seed_value)

    # Splitting data into train and test based on ratio
    train_data, test_data = train_test_split(data, test_size=1 - ratio, random_state=seed_value)

    # Subsetting train data to balance good and bad observations
    good_obs = train_data[train_data[default_flag] == '0'].sample(n=balance_count, replace=True)
    bad_obs = train_data[train_data[default_flag] == '1'].sample(n=balance_count, replace=True)

    # Combining balanced samples
    train_balanced_data = pd.concat([good_obs, bad_obs])

    return {'train_balanced': train_balanced_data, 'test': test_data}

