from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd

def Gini_univariate_data(data, default_flag):
    """
    Calculate the Gini coefficient from the estimated values calculated by logistic regression for each variable in the dataset.

    Parameters:
    - data (DataFrame): The dataset.
    - default_flag (str): The name of the default flag variable.

    Returns:
    DataFrame: DataFrame containing variables and their corresponding Gini values.
    """
    gini_values = []
    variable_names = []

    for column in data.columns:
        if column != default_flag:
            X = data[[column]]
            y = data[default_flag]

            # Fit logistic regression model
            model = LogisticRegression(solver='liblinear')
            model.fit(X, y)

            # Predict probabilities
            y_pred = model.predict_proba(X)[:, 1]

            # Calculate Gini coefficient
            gini_value = 2 * roc_auc_score(y, y_pred) - 1

            # Append variable name and Gini value to lists
            variable_names.append(column)
            gini_values.append(gini_value)

    # Create DataFrame from lists
    gini_df = pd.DataFrame({'Variable': variable_names, 'Gini': gini_values})

    # Sort DataFrame by Gini values in descending order
    ordered_gini_df = gini_df.sort_values(by='Gini', ascending=False)

    # Reset index
    ordered_gini_df.reset_index(drop=True, inplace=True)

    return ordered_gini_df

