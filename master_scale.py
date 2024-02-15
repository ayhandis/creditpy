import pandas as pd
import numpy as np

def master_scale(data, default_flag, PD, bin_number):
    """
    Master Scale

    This function creates a master scale that best describes the target variable according to the given parameters.

    Parameters:
    data : pandas DataFrame
        The dataset.
    default_flag : str
        The column name of the default flag.
    PD : str
        The column name of the PD variable.
    bin_number : int
        The number of bins to create for binning the PD variable.
    stop_limit : float, optional
        Stops binning of the predictor's classes/levels in case the resulting information value (IV)
        decreases more than x percent (e.g. 0.05 = 5 percent) compared to the preceding binning step.
        Accepted range: 0-0.5; default: 0.1.
    min_perc_total : float, optional
        For numeric variables, this parameter defines the number of initial classes before any merging is applied.
        Accepted range: 0.0001-0.2; default: 0.05.
    min_perc_class : float, optional
        If a column percentage of one of the target classes within a bin is below this limit (e.g. below 0.01=1 percent),
        then the respective bin will be joined with others.
        Accepted range: 0-0.2; default: 0, i.e. no merging with respect to sparse target classes is applied.

    Returns:
    pandas DataFrame
        A DataFrame containing the master scale with PD and Score columns.

    Examples:
    default_f = ['1','0','0', '1','1','0','0','1','1']
    probability = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    example_data = pd.DataFrame({'default_f': default_f, 'Probability': probability})
    master_scale(example_data, "default_f", "Probability", bin_number=10)
    """
    # Implement custom binning logic here
    data['PD_Bin'] = pd.cut(data[PD], bins=bin_number)

    # Aggregate data based on PD bins
    summary = data.groupby('PD_Bin').agg({
        default_flag: ['count', 'sum'],
        PD: ['mean', 'std']
    }).reset_index()

    # Calculate WOE and Score using custom logic
    summary['Good.Count'] = summary[(default_flag, 'count')] - summary[(default_flag, 'sum')]
    summary['Bad.Count'] = summary[(default_flag, 'sum')]
    summary['Total.Observations'] = summary[(default_flag, 'count')]
    summary['Good.Distr'] = summary['Good.Count'] / len(data)
    summary['Bad.Distr'] = summary['Bad.Count'] / len(data)
    summary['Total.Distr'] = summary['Total.Observations'] / len(data)
    summary['Bad.Rate'] = summary['Bad.Count'] / summary['Total.Observations']
    summary['PD'] = summary[(PD, 'mean')]
    summary['Score'] = -np.log(summary['PD', 'mean'] / (1 - summary['PD', 'mean'])) * 100

    # Select and rename columns
    woe_summary = summary[['PD_Bin', 'Total.Observations', 'Total.Distr', 'Good.Count', 'Bad.Count',
                           'Good.Distr', 'Bad.Distr', 'Bad.Rate', 'PD', 'Score']]
    woe_summary.columns = ['Final.PD.Range', 'Total.Observations', 'Total.Distr', 'Good.Count', 'Bad.Count',
                           'Good.Distr', 'Bad.Distr', 'Bad.Rate', 'PD','PD_2', 'Score']

    woe_summary.drop(columns=['PD_2'], inplace=True)

    return woe_summary
