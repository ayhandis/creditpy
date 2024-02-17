import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_calc(X):
    """
    Calculate Variance Inflation Factor (VIF) for a set of predictor variables.

    VIF measures the multicollinearity among predictor variables in a regression model.
    High VIF values indicate high multicollinearity.

    Parameters:
    X : pandas DataFrame
        The design matrix containing the predictor variables.

    Returns:
    pandas DataFrame:
        VIF values for each predictor variable.
    """
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.set_index("Variable")["VIF"]


