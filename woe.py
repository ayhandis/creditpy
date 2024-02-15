import pandas as pd
import numpy as np

def woe_binning(df_train, df_test, target_column, bins=10):
    """
    Apply WOE transformation to the specified training and test dataframes.

    Parameters:
        df_train : DataFrame
            The training dataframe containing the predictor variables.
        df_test : DataFrame
            The test dataframe containing the predictor variables.
        target_column : str
            The name of the target column in the dataframes.
        bins : int, optional (default=10)
            The number of bins to use for binning the predictor variables.

    Returns:
        train_woe : DataFrame
            The training dataframe with WOE-transformed variables.
        test_woe : DataFrame
            The test dataframe with WOE-transformed variables.
    """
    def calculate_woe(df, column, target_column):
        grouped = df.groupby(column)[target_column].agg(['count', 'sum'])
        grouped['non_events'] = grouped['count'] - grouped['sum']
        grouped['events'] = grouped['sum']
        grouped['event_rate'] = grouped['events'] / grouped['events'].sum()
        grouped['non_event_rate'] = grouped['non_events'] / grouped['non_events'].sum()
        grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
        woe_values = grouped['woe'].to_dict()
        return woe_values

    def apply_woe(df, woerules):
        for column, rule in woerules.items():
            bin_column = column + '_bin'
            if bin_column in df.columns:
                df[bin_column] = df[bin_column].map(rule)
                df.drop(column, axis=1, inplace=True)  # Drop the original column
            else:
                print(f"Binning column '{bin_column}' not found in the dataframe.")
        return df

    woerules = {}
    for column in df_train.columns:
        if column != target_column:
            if not pd.api.types.is_numeric_dtype(df_train[column]):
                try:
                    df_train[column] = pd.to_numeric(df_train[column])
                    df_test[column] = pd.to_numeric(df_test[column])
                except ValueError:
                    print(f"Column '{column}' could not be converted to numeric type.")
                    continue
            binned_column, bins_info = pd.qcut(df_train[column], bins, retbins=True, duplicates='drop')
            df_train[column + '_bin'] = binned_column
            df_test[column + '_bin'] = pd.cut(df_test[column], bins=bins_info, labels=bins_info[:-1], include_lowest=True)
            woerules[column] = calculate_woe(df_train, column + '_bin', target_column)

    train_woe = apply_woe(df_train, woerules)
    test_woe = apply_woe(df_test, woerules)

    # Convert all columns to numeric
    train_woe = train_woe.apply(pd.to_numeric, errors='ignore')
    test_woe = test_woe.apply(pd.to_numeric, errors='ignore')

    # Replace infinite values with 0
    train_woe.replace([np.inf, -np.inf], 0, inplace=True)
    test_woe.replace([np.inf, -np.inf], 0, inplace=True)

    return train_woe, test_woe
