import pandas as pd

def Anchor_point(master_scale_data, PD, total_observations, central_tendency, upper_green=1.2, upper_red=1.3, lower_green=0.8, lower_red=0.7):
    """
    Perform Anchor Point test for a given master scale data.

    Parameters:
    - master_scale_data (DataFrame): Master scale data set.
    - PD (str): PD variable.
    - total_observations (str): Total observations variable.
    - central_tendency (float): Central tendency for testing.
    - upper_green (float): Upper green threshold, default is 1.2.
    - upper_red (float): Upper red threshold, default is 1.3.
    - lower_green (float): Lower green threshold, default is 0.8.
    - lower_red (float): Lower red threshold, default is 0.7.

    Returns:
    DataFrame: DataFrame containing test results.
    """
    concentration = master_scale_data[total_observations] / master_scale_data[total_observations].sum()
    avg_pd = (master_scale_data[PD] * master_scale_data[total_observations]).sum() / master_scale_data[total_observations].sum()
    upper_g = avg_pd * upper_green
    upper_r = avg_pd * upper_red
    lower_g = avg_pd * lower_green
    lower_r = avg_pd * lower_red

    data = pd.DataFrame({'central_tendency': [central_tendency],
                         'avg_pd': [avg_pd],
                         'lower_r': [lower_r],
                         'lower_g': [lower_g],
                         'upper_g': [upper_g],
                         'upper_r': [upper_r]})

    data['test_result'] = data.apply(lambda row: 'Green' if (row['central_tendency'] > row['lower_g'] and row['central_tendency'] < row['upper_g']) else ('Red' if (row['central_tendency'] < row['lower_r'] or row['central_tendency'] > row['upper_r']) else 'Yellow'), axis=1)

    return data
