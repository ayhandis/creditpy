def Herfindahl_Hirschman_Index(data, total_observations):
    """
    Calculate the Herfindahl-Hirschman Index (HHI) for master scales.

    Parameters:
    - data (DataFrame): The dataset.
    - total_observations (str): The name of the total observations variable.

    Returns:
    float: The calculated HHI.
    """
    sum_total = data[total_observations].sum()
    data['concentration'] = data[total_observations] / sum_total
    data['HHI'] = data['concentration'] ** 2
    print(data)
    return data['HHI'].sum()

