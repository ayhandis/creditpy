import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def k_fold_cross_validation_glm(model_data, default_flag, folds, seed_value):
    """
    K Fold Cross Validation Gini

    This function creates k fold cross-validation data sets and calculates Gini coefficient for logistic regression.

    Parameters:
    model_data : pandas DataFrame
        The dataset.
    default_flag : str
        The column name of the default flag.
    folds : int
        The number of folds desired.
    seed_value : int
        A seed value for replicability.

    Returns:
    pandas DataFrame
        A DataFrame containing Gini coefficients for each fold and their averages.

    Examples:
    default_f = ['1','0','0', '1','1','0','0','1','1']
    birth_year = [1980, 1985, 1971, 1971, 1985, 1971, 1980, 1980, 1985]
    job = [1, 1, 2, 2, 2, 3, 3, 2, 3]
    example_data = pd.DataFrame({'default_f': default_f, 'birth_year': birth_year, 'job': job})
    k_fold_cross_validation_glm(example_data, "default_f", 10, 1)
    """
    np.random.seed(seed_value)
    # Randomly shuffle the data
    model_data = model_data.sample(frac=1).reset_index(drop=True)

    # Drop non-numeric columns
    numeric_columns = model_data.select_dtypes(include=[np.number]).columns
    model_data = model_data[numeric_columns]

    # Initialize arrays to store Gini coefficients
    gini_fold_train = []
    gini_fold_test = []

    # Perform k-fold cross-validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed_value)
    for train_index, test_index in skf.split(model_data.drop(columns=[default_flag]), model_data[default_flag]):
        train_data, test_data = model_data.iloc[train_index], model_data.iloc[test_index]

        # Fit logistic regression model
        model = LogisticRegression()
        model.fit(train_data.drop(columns=[default_flag]), train_data[default_flag])

        # Predict probabilities
        train_pred_prob = model.predict_proba(train_data.drop(columns=[default_flag]))[:, 1]
        test_pred_prob = model.predict_proba(test_data.drop(columns=[default_flag]))[:, 1]

        # Calculate Gini coefficients
        gini_train = 2 * roc_auc_score(train_data[default_flag], train_pred_prob) - 1
        gini_test = 2 * roc_auc_score(test_data[default_flag], test_pred_prob) - 1

        # Append Gini coefficients
        gini_fold_train.append(gini_train)
        gini_fold_test.append(gini_test)

    # Calculate average Gini coefficients
    avg_gini_train = np.mean(gini_fold_train)
    avg_gini_test = np.mean(gini_fold_test)

    # Create DataFrame to store results
    fold_result = pd.DataFrame({
        'Fold': range(1, folds + 1),
        'GiniTrain': gini_fold_train,
        'GiniTest': gini_fold_test
    })
    fold_result.loc[len(fold_result)] = ['Average', avg_gini_train, avg_gini_test]

    return fold_result
