import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def max_gini_model(data, default_flag, seed_value=1):
    """
    Maximum Gini Model

    This function finds the model which gives the maximum Gini value. Statistical requirements will not be provided.
    Can only be used to give an inference.

    Parameters:
    data : pandas DataFrame
        The dataset.
    default_flag : str
        The column name of the default flag.
    seed_value : int, optional
        A seed value for replicability. Default is 1.

    Returns:
    sklearn.linear_model._logistic.LogisticRegression
        The logistic regression model with the maximum Gini value.

    Examples:
    default_f = ['1','0','0', '1','1','0','0','1','1']
    birth_year = [1980, 1985, 1971, 1971, 1985, 1971, 1980, 1980, 1985]
    job = [1, 1, 2, 2, 2, 3, 3, 2, 3]
    example_data = pd.DataFrame({'default_f': default_f, 'birth_year': birth_year, 'job': job})
    max_gini_model(example_data, "default_f", 10)
    """
    # Get all combinations of predictor variables
    predictor_cols = [col for col in data.columns if col != default_flag]
    models = []
    max_gini = -1
    best_model = None
    for r in range(1, len(predictor_cols) + 1):
        for comb in combinations(predictor_cols, r):
            formula = f"{default_flag} ~ {' + '.join(comb)}"
            model = LogisticRegression()
            model.fit(data[list(comb)], data[default_flag])
            y_pred_prob = model.predict_proba(data[list(comb)])[:, 1]
            gini = 2 * roc_auc_score(data[default_flag], y_pred_prob) - 1
            print(f"Formula: {formula}, Gini: {gini}")
            if gini > max_gini:
                max_gini = gini
                best_model = model
    print(f"Best Model Gini: {max_gini}")
    return best_model


