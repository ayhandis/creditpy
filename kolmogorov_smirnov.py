from scipy.stats import ks_2samp

def Kolmogorov_Smirnov(data, default_flag, PD):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for a given scoring data.

    Parameters:
    - data (DataFrame): The dataset.
    - default_flag (str): The name of the default flag variable.
    - PD (str): The name of the PD variable.

    Returns:
    KS_Result: A tuple containing the KS statistic and the p-value.
    """
    events = data[data[default_flag] == 1][PD]
    non_events = data[data[default_flag] == 0][PD]

    ks_statistic, ks_pvalue = ks_2samp(events, non_events)

    KS_Result = {
        "KS Statistic (%)": ks_statistic * 100,
        "P-Value": ks_pvalue
    }

    return KS_Result


