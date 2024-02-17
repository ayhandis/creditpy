import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def regression_calibration(model, calibration_data, default_flag):
    """
    Perform model calibration for a logistic regression model.

    This function takes a logistic regression model, a calibration dataset,
    and the name of the default flag variable. It calibrates the model using
    logistic regression on the provided calibration dataset.

    Parameters:
    model (sklearn.linear_model.LogisticRegression): The logistic regression model to be calibrated.
    calibration_data (DataFrame): The calibration dataset containing features and the default flag variable.
    default_flag (str): The name of the default flag variable in the calibration dataset.

    Returns:
    dict: A dictionary containing calibrated probabilities, calibrated scores, and calibration formula.

    Example:
    >>> from sklearn.linear_model import LogisticRegression
    >>> import pandas as pd
    >>> # Assume model and calibration_data are already defined
    >>> calibration_result = regression_calibration(model, calibration_data, "default_flag")
    >>> print(calibration_result['calibration_data'])
    >>> print(calibration_result['calibration_model'])
    >>> print(calibration_result['calibration_formula'])
    """

    # Calculate model predicted probabilities
    calibration_data['modelpd'] = model.predict_proba(calibration_data.drop(default_flag, axis=1))[:, 1]

    # Calculate model scores
    calibration_data['modelscore'] = np.log(calibration_data['modelpd'] / (1 - calibration_data['modelpd']))

    # Fit calibration model
    calibration_model = LogisticRegression()
    calibration_model.fit(calibration_data[['modelscore']], calibration_data[default_flag])

    # Calculate calibrated probabilities
    calibration_data['calibrated_pd'] = calibration_model.predict_proba(calibration_data[['modelscore']])[:, 1]

    # Calculate calibrated scores
    calibration_data['calibrated_score'] = np.log(calibration_data['calibrated_pd'] / (1 - calibration_data['calibrated_pd']))

    # Prepare formula strings
    var_names = calibration_data.columns[:-4]  # Exclude modelpd, modelscore, calibrated_pd, calibrated_score
    equation_str = " + ".join([f"{coef:.7f}*{var_name}" for coef, var_name in zip(model.coef_[0], var_names)])
    equation_str = f"{equation_str} + {model.intercept_[0]}"
    equation_str = f"{model.coef_[0]}*{var_names} + {model.intercept_[0]}"

    var_names_cal = calibration_data.columns[:-6]  # Exclude modelpd, modelscore, calibrated_pd, calibrated_score, default_flag
    equation_str_cal = " + ".join([f"{coef:.7f}*{var_name}" for coef, var_name in zip(calibration_model.coef_[0], var_names_cal)])
    equation_str_cal = f"{equation_str_cal} + {calibration_model.intercept_[0]}"

    # Prepare calibration formula
    calibration_formula = f"modelscore formula ::: {equation_str}, calibrated_score formula ::: {equation_str_cal}, Calibration Formula to get calibrated_pd ::: 1/(1 + exp(-calibrated_score))"

    return {
        'calibration_data': calibration_data,
        'calibration_model': calibration_model,
        'calibration_formula': calibration_formula
    }
