import pandas as pd
import numpy as np

def woe_glm_feature_importance(model_data, model, default_flag):
    """
    Calculate feature importance for a given logistic regression model with WOE transformation method.

    Parameters:
    model_data : pandas DataFrame
        The dataset used to build the logistic regression model.
    model : sklearn.linear_model.LogisticRegression
        The logistic regression model built using WOE transformation.
    default_flag : str
        The name of the default flag column in the dataset.

    Returns:
    pandas DataFrame:
        DataFrame containing variable names and their respective feature importance.

    Example:
    >>> # Assume model_data and model are defined
    >>> importance = woe_glm_feature_importance(model_data_v1, my_model, "default_f")
    >>> print(importance)
    """

    if not isinstance(model_data, pd.DataFrame):
        raise ValueError("Input 'model_data' must be a pandas DataFrame.")

    if default_flag not in model_data.columns:
        raise ValueError("Column '{}' not found in the dataset.".format(default_flag))

    # Extract coefficient values from the model
    coef = model.coef_[0]

    # Calculate feature importance
    model_importance_std = []
    model_importance_colnames = []

    for column, col_coef in zip(model_data.columns, coef):
        if column != default_flag:
            col_std = np.std(model_data[column])
            model_importance_std.append(col_coef * col_std)
            model_importance_colnames.append(column)

    # Normalize feature importance
    importance = np.array(model_importance_std) / np.sum(model_importance_std)

    # Create DataFrame with variable names and feature importance
    return_data = pd.DataFrame({
        'Variables': model_importance_colnames,
        'beta_std': model_importance_std,
        'importance': importance
    })

    return return_data
