import pandas as pd
import numpy as np

def train_test_split(data, seed_value=1, ratio=0.67):
    """
    Split the dataset into train and test sets.

    This function separates the dataset into train and test sets based on the provided seed value and ratio.

    Parameters:
    data (DataFrame): The dataset to be split.
    seed_value (int): Seed value for randomization (default is 1).
    ratio (float): The percentage of data to be used for the train set (default is 0.67).

    Returns:
    dict: A dictionary containing the train and test sets.
    #
    # Example:
    # >>> import pandas as pd
    # >>> # Assume data is defined
    # >>> random_column = pd.DataFrame({'random_column': np.random.uniform(0, 1000, 100)})
    # >>> datasets = train_test_split(random_column, seed_value=1, ratio=0.70)
    # >>> train = datasets['train']
    # >>> test = datasets['test']
    # """

    np.random.seed(seed_value)

    # Randomly shuffle the indices
    indices = np.random.permutation(data.index)

    # Determine the split index based on the ratio
    split_index = int(len(data) * ratio)

    # Split the data into train and test sets
    train_data = data.iloc[indices[:split_index]]
    test_data = data.iloc[indices[split_index:]]

    return {'train': train_data, 'test': test_data}

