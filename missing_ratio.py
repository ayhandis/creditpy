import pandas as pd


def missing_ratio(data):
    """
    Missing Ratio

    This function calculates the missing ratios of variables for a given data set.

    Parameters:
    data : pandas DataFrame
        The dataset.

    Returns:
    pandas DataFrame
        A DataFrame containing the variable names and their corresponding missing ratios.

    Examples:
    name = ['John Doe', 'Peter Gynn', 'Jolie Hope']
    birth_year = [1980, 1985, 1971]
    salary = [20000, None, 10000]
    example_data = pd.DataFrame({'name': name, 'birth_year': birth_year, 'salary': salary})
    missing_ratio(example_data)
    """
    missing_ratios = data.isnull().mean()
    completeness = 1 - missing_ratios
    missing_info = pd.DataFrame(
        {'Variable': missing_ratios.index, 'Missing_Ratio': missing_ratios.values, 'Completeness': completeness.values})

    return missing_info

