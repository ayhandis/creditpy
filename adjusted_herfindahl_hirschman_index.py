def Adjusted_Herfindahl_Hirschman_Index(data, total_observations):
    """
    Calculate Adjusted Herfindahl-Hirschman Index (HHI) for master scales.

    Parameters:
    - data (DataFrame): The dataset.
    - total_observations (str): The column name for the total observations variable.

    Returns:
    float: Adjusted HHI value.
    """

    data['SumTotal'] = data[total_observations].sum()
    data['concentration'] = data[total_observations] / data['SumTotal']
    data['HHI'] = data['concentration'] ** 2

    HI = data['HHI'].sum()
    AdjustedHHI = (HI - 1 / len(data['SumTotal'])) / (1 - 1 / len(data['SumTotal']))

    return AdjustedHHI
