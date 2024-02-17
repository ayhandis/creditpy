import pandas as pd

def summary_default_flag(data, default_flag, variable):
    """
    Calculate summary statistics of a variable based on good and bad observations.

    This function calculates summary statistics (mean, median, min, max, etc.)
    of a given variable based on good and bad observations defined by the default flag.

    Parameters:
    data (DataFrame): The dataset containing the variable and default flag.
    default_flag (str): The name of the default flag variable in the dataset.
    variable (str): The name of the variable for which summary statistics are calculated.

    Returns:
    None

    Example:
    # >>> import pandas as pd
    # >>> # Assume data is defined
    # >>> default_f = ['1', '0', '0', '1']
    # >>> birth_year = [1980, 1985, 1971, 1990]
    # >>> salary = [20000, None, 10000, 10050]
    # >>> example_data = pd.DataFrame({'default_f': default_f, 'birth_year': birth_year, 'salary': salary})
    # >>> summary_default_flag(example_data, "default_f", "birth_year")
    # """
    good_obs = data[data[default_flag] == '0'][variable]
    bad_obs = data[data[default_flag] == '1'][variable]

    print("Good Observations")
    print(good_obs.describe())
    print("\nBad Observations")
    print(bad_obs.describe())

