import pandas as pd
import numpy as np
from calculate_gini import calculate_gini
from missing_ratio import missing_ratio
from sklearn.model_selection import train_test_split
from woe import woe_binning
from iv_calc import IV_calc
from iv_calc_data import IV_calc_data
from gini_univariate_data import Gini_univariate_data
from gini_univariate import Gini_univariate
from gini_elimination import Gini_elimination
from variable_clustering import variable_clustering
from variable_clustering_gini import variable_clustering_gini
from correlation_cluster import correlation_cluster
from max_gini_model import max_gini_model
from woe_glm_feature_importance import woe_glm_feature_importance
from scaled_score import scaled_score
from regression_calibration import regression_calibration
from master_scale import master_scale
from bayesian_calibration import bayesian_calibration
from vif_calc import vif_calc
from kfold_cross_validation_glm import k_fold_cross_validation_glm
from kolmogorov_smirnov import Kolmogorov_Smirnov
from ssi_calc_data import SSI_calc_data
from psi_calc_data import PSI_calc_data
from herfindahl_hirschman_index import Herfindahl_Hirschman_Index
from anchor_point import Anchor_point
from chisquare_test import chisquare_test
from binomial_test import Binomial_test


# Load sample data
germancredit = pd.read_csv('data/german_credit.csv')

# Preparing sample data
sample_data = germancredit[["duration.in.month", "credit.amount", "installment.rate.in.percentage.of.disposable.income",
                            "age.in.years", "creditability"]]

# # Convert 'creditability' (default flag) variable into numeric type
# sample_data["creditability"] = sample_data["creditability"].apply(lambda x: 1 if x == "bad" else 0)

# Calculating missing ratios
missing_ratio_result = missing_ratio(sample_data)
print("Missing Ratio:", missing_ratio_result)

# Splitting data into train and test sets
train, test = train_test_split(sample_data, random_state=123, train_size=0.70)

# Applying WOE transformation
train_woe, test_woe = woe_binning(train, test, target_column='creditability')


# Calculate IV and Gini for the whole dataset
IV_summary = IV_calc_data(train_woe, "creditability")
print("Information Value (IV) Summary:", IV_summary)
gini_summary = Gini_univariate_data(train_woe, "creditability")
print("Univariate Gini Summary:", gini_summary)

# Gini elimination
eliminated_data = Gini_elimination(train_woe, "creditability", 0.1825)
print("Data after Gini elimination:\n", eliminated_data)

# Variable clustering
clustering_data = variable_clustering(eliminated_data, "creditability", 2)
print("Variable Clustering Data:", clustering_data)
gini_values = variable_clustering_gini(eliminated_data, "creditability", 2)
print("Gini Values:", gini_values)
# Call the correlation_cluster function
correlation_cluster_result = correlation_cluster(eliminated_data, clustering_data, clusters='Group', target_column = 'creditability')
print("Correlation Cluster Result:", correlation_cluster_result)

# Logistic regression model
model = max_gini_model(eliminated_data, "creditability", 10)

# Calculate variable weights
variable_weights = woe_glm_feature_importance(eliminated_data, model, "creditability")
print("Variable Weights:", variable_weights)


# Get the columns used for training the model (excluding the target variable)
training_columns = eliminated_data.drop(columns=['creditability']).columns
# Fill missing values with 0 in the training data (This is just for example usage)
eliminated_data.fillna(0, inplace=True)  # Replace NaN with 0
# Generate PD values for train data using aligned columns
train_probs = model.predict_proba(eliminated_data[training_columns])[:, 1]
ms_train_data = pd.concat([eliminated_data[training_columns], pd.Series(train_probs, name="PD", index=eliminated_data.index)], axis=1)

# Align the columns of the test dataset with the training columns
test_data_aligned = test_woe[training_columns]
# Fill missing values with 0 (This is just for example usage)
test_data_aligned.fillna(0, inplace=True)  # Replace NaN with 0
# Generate PD values for test data using aligned columns
test_probs = model.predict_proba(test_data_aligned)[:, 1]
ms_test_data = pd.concat([test_data_aligned, pd.Series(test_probs, name="PD", index=test_data_aligned.index)], axis=1)
ms_train_data['creditability'] = eliminated_data['creditability']
ms_test_data['creditability'] = test_woe['creditability']


# Bayesian calibration
ms_train_data["Score"] = np.log(ms_train_data["PD"] / (1 - ms_train_data["PD"]))
ms_test_data["Score"] = np.log(ms_test_data["PD"] / (1 - ms_test_data["PD"]))
master_scale_data = master_scale(ms_train_data, "creditability", "PD",10)
bayesian_method = bayesian_calibration(master_scale(ms_train_data, "creditability", "PD",10), average_score='Score', calibration_data = ms_train_data, calibration_data_score="Score", total_observations= 'Total.Observations', PD = "PD", central_tendency=0.05)
print("Calibration model:", bayesian_method["Calibration_model"].summary())
print("Calibration formula:", bayesian_method["Calibration_formula"])
print("Master scale data:", bayesian_method["Data"].head())
print("Calibration data:", bayesian_method["Calibration_data"].head())

# Scaled score
scaled_score_data = scaled_score(bayesian_method["Calibration_data"], "calibrated_pd", 3000, 15)
print("Scaled Score Data:", scaled_score_data)

# Calculate VIF
vif_values = vif_calc(eliminated_data)
print("VIF Values:", vif_values)

# Assuming you have predictions and actual values from your model
predictions = ms_test_data['PD']
actual_values = ms_test_data["creditability"]

# Calculate Gini coefficient for the model
gini_value = calculate_gini(predictions, actual_values)
print("Gini Value:", gini_value)

# 5 Fold cross-validation
k_fold_result = k_fold_cross_validation_glm(ms_train_data, "creditability", 5, 1)
print("5 Fold Cross Validation Result:", k_fold_result)

# KS test
ks_result_train = Kolmogorov_Smirnov(ms_train_data, "creditability", "PD")
print("KS Result (Train Data):", ks_result_train)
ks_result_test = Kolmogorov_Smirnov(ms_test_data, "creditability", "PD")
print("KS Result (Test Data):", ks_result_test)

# Variable stabilities measurement
psi_result = PSI_calc_data(train_woe, test_woe, bins = 10, default_flag="creditability")
print("PSI Result:", psi_result)

# HHI test
hhi_value = Herfindahl_Hirschman_Index(master_scale_data,"Total.Observations")
print("HHI Value:", hhi_value)

# Anchor point test
anchor_result = Anchor_point(master_scale_data, "PD", "Total.Observations", 0.30)
print("Anchor Point Result:", anchor_result)

# Chi-square test
chisquare_result = chisquare_test(master_scale_data, "PD", "Bad.Count", "Total.Observations", 0.90)
print("Chi-square Test Result:", chisquare_result)

# Binomial test
binomial_result = Binomial_test(master_scale_data, "Total.Observations", "PD", "Bad.Rate", 0.90, "one")
print("Binomial Test Result:", binomial_result)
