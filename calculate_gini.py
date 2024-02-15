from sklearn.metrics import roc_auc_score

def calculate_gini(predictions, actual):
    """
    Calculate Gini coefficient for a model using ROC curve.

    Parameters:
    predictions : array-like
        Predicted values from the model.
    actual : array-like
        Actual values from the test data.

    Returns:
    float:
        Gini coefficient value.
    """
    # Calculate the Area Under the ROC Curve (AUC)
    auc = roc_auc_score(actual, predictions)

    # Calculate Gini coefficient from AUC
    gini_coefficient = (2 * auc) - 1

    return gini_coefficient