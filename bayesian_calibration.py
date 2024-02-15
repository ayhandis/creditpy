import pandas as pd
import numpy as np
import statsmodels.api as sm

def bayesian_calibration(data, average_score, total_observations, PD, central_tendency, calibration_data, calibration_data_score):
    """
    Perform Bayesian calibration for a model using the Bayesian method.

    Parameters:
    - data (pandas.DataFrame): The master scale data.
    - average_score (str): Name of the average score variable in the master scale data.
    - total_observations (str): Name of the total observations variable in the master scale data.
    - PD (str): Name of the PD variable in the master scale data.
    - central_tendency (float): The central tendency (calibration target).
    - calibration_data (pandas.DataFrame): The scoring model data.
    - calibration_data_score (str): Name of the score variable in the calibration data.

    Returns:
    dict: A dictionary containing the calibration model, calibration formula, master scale data, and calibration data.
    """
    # Calculate average PD
    avg_pd = (data[total_observations] * data[PD]).sum() / data[total_observations].sum()

    # Calculate calibrated PD
    data['calibrated_pd'] = (data[PD] * central_tendency / avg_pd) / \
                             (data[PD] * central_tendency / avg_pd + (1 - data[PD]) * ((1 - central_tendency) / (1 - avg_pd)))

    # Calculate odds ratio
    data['OddRatio'] = (1 - data['calibrated_pd']) / data['calibrated_pd']

    # Fit calibration model
    X = sm.add_constant(data[average_score])
    y = np.log(data['OddRatio'])
    calibration_model = sm.OLS(y, X).fit()

    # Extract coefficients
    intercept = calibration_model.params[0]
    score_coef = calibration_model.params[1]

    # Apply calibration to calibration data
    calibration_data['calibrated_pd'] = 1 / (1 + np.exp(intercept + calibration_data[calibration_data_score] * score_coef))

    # Construct calibration information dictionary
    calibration_info = {
        'Calibration_model': calibration_model,
        'Calibration_formula': f"Calibration method can be applied with: 1/(1+exp(Intercept + Score * Coefficient)) formula. Numerically: 1/(1+exp({intercept} + Score * {score_coef}))",
        'Data': data,
        'Calibration_data': calibration_data
    }

    return calibration_info

