import pandas as pd


def na_checker(data):
    """
    NA Checker

    This function checks the presence of NA values in variables for a given data set.
    For the results returned as True, the corresponding variable has NA observation/observations.

    Parameters:
    data : pandas DataFrame
        The dataset.

    Returns:
    pandas Series
        A Series indicating whether each variable has NA values (True) or not (False).

    Examples:
    name = ['John Doe', 'Peter Gynn', 'Jolie Hope']
    birth_year = [1980, 1985, 1971]
    salary = [20000, None, 10000]
    example_data = pd.DataFrame({'name': name, 'birth_year': birth_year, 'salary': salary})
    na_checker(example_data)
    """
    return data.isnull().any()

