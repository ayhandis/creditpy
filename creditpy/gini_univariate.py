from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc

def Gini_univariate(data, default_flag, variable):
    """
    Calculate the univariate Gini coefficient from the estimated values calculated by logistic regression of a variable.

    Parameters:
    - data (DataFrame): The dataset.
    - default_flag (str): The name of the default flag variable.
    - variable (str): The name of the variable for which the Gini value is to be calculated.

    Returns:
    float: Univariate Gini value.
    """
    X = data[[variable]]
    y = data[default_flag]

    # Fit logistic regression model
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)

    # Predict probabilities
    y_pred = model.predict_proba(X)[:, 1]

    # Calculate Gini coefficient
    univ_gini = 2 * roc_auc_score(y, y_pred) - 1

    return univ_gini

