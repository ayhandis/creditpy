import pandas as pd


def missing_elimination(data, missing_ratio_threshold):
    """
    Missing Elimination

    This function eliminates variables which have missing ratios greater than a given threshold for a given data set.

    Parameters:
    data : pandas DataFrame
        The dataset.
    missing_ratio_threshold : float
        The missing ratio threshold. Variables with missing ratios greater than this threshold will be eliminated.

    Returns:
    pandas DataFrame
        The modified dataset with variables eliminated based on the missing ratio threshold.

    Examples:
    name = ['John Doe', 'Peter Gynn', 'Jolie Hope']
    birth_year = [1980, 1985, 1971]
    salary = [20000, None, 10000]
    example_data = pd.DataFrame({'name': name, 'birth_year': birth_year, 'salary': salary})
    missing_elimination(example_data, 0.10)
    """
    missing_ratios = data.isnull().mean()
    eliminated_columns = missing_ratios[missing_ratios > missing_ratio_threshold].index
    eliminated_data = data.drop(columns=eliminated_columns)

    return eliminated_data

