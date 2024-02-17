# creditpy/data_loader.py
import os
import pandas as pd

def load_german_credit_data():
    """
    Load the german_credit.csv file from the data folder under the creditpy package.

    Returns:
    DataFrame: DataFrame containing the loaded data.
    """
    # Get the path to the data folder within the creditpy package
    package_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(package_dir, 'data')

    # Construct the path to the german_credit.csv file
    file_path = os.path.join(data_dir, 'german_credit.csv')

    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    return data
