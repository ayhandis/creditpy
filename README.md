# CreditPy
## A Credit Risk Scoring and Validation Package

CreditPy is a Python package developed as a successor to the CreditR package, drawing inspiration from its successful history. With a wide user base including companies in sectors such as audit and consultancy firms, technology consultancy firms, management consultancy firms, banks, financial institutions, and fintech companies, CreditPy aims to provide an easy model set-up functionality in the field of credit risk scoring.

CreditPy enables the efficient implementation of credit risk scoring methodologies. It facilitates tasks such as variable analysis, variable selection, model development, model calibration, rating scale development, and model validation quickly and effectively.

CreditPy continues the legacy of CreditR, which has had a successful global adoption and integration across various industries. It is used by companies from Turkey, the Netherlands, the United States, India, the United Kingdom, Nigeria, South Africa, Australia, Brazil, Mexico, Belgium, Sweden, Dubai, Abu Dhabi, Saudi Arabia, Azerbaijan, Russia and many others. Designed with an intellectual perspective, CreditPy aims to cover the needs of basic credit risk scoring applications.

### Prerequisites
Before using CreditPy, please make sure you have Python installed on your system. You can install CreditPy and its dependencies using pip:
```
pip install creditpy
```

### Getting Started
You can import CreditPy modules and use its functions as follows:
```
import pandas as pd
from creditpy import calculate_gini, missing_ratio, train_test_split, woe, IV_calc_data, Gini_univariate_data, Gini_elimination, variable_clustering, variable_clustering_gini, correlation_cluster, max_gini_model, woe_glm_feature_importance, scaled_score, regression_calibration, master_scale, bayesian_calibration, vif_calc, k_fold_cross_validation_glm, Kolmogorov_Smirnov, PSI_calc_data, Herfindahl_Hirschman_Index, Anchor_point, chisquare_test, Binomial_test

```

### An Application of the Package
An example application of the package is shared below in a study of how some common steps in credit risk scoring are carried out using the functions provided in the package.

```
#This Python script is designed to make the creditpy package easier to understand.
#Obtaining a high accuracy model is not within the scope of this study.

# Load sample data
germancredit = pd.read_csv('data/german_credit.csv')

# Prepare sample data
sample_data = germancredit[["duration.in.month", "credit.amount", "installment.rate.in.percentage.of.disposable.income",
                            "age.in.years", "creditability"]]

# Calculate missing ratios
missing_ratio_result = missing_ratio(sample_data)
print("Missing Ratio:", missing_ratio_result)

# Split data into train and test sets
train, test = train_test_split(sample_data, random_state=123, train_size=0.70)

# Apply WOE transformation
train_woe, test_woe = woe(train, test, target_column='creditability')

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
correlation_cluster_result = correlation_cluster(eliminated_data, clustering_data, clusters='Group', target_column="creditability")
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
master_scale_data = master_scale(ms_train_data, "creditability", "PD", 10)
bayesian_method = bayesian_calibration(master_scale(ms_train_data, "creditability", "PD", 10), average_score='Score', calibration_data = ms_train_data, calibration_data_score="Score", total_observations= 'Total.Observations', PD = "PD", central_tendency=0.05)
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
psi_result = PSI_calc_data(train_woe, test_woe, bins=10, default_flag="creditability")
print("PSI Result:", psi_result)

# HHI test
hhi_value = Herfindahl_Hirschman_Index(master_scale_data, "Total.Observations")
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
```

## Bug Fixes

Please inform me about the errors you have encountered while using the package via the e-mail address that is shared in the Author section.

## Author

* **Ayhan Dis**  - [Github](https://github.com/ayhandis) - [Linkedin](https://www.linkedin.com/in/ayhandis/)  - disayhan@gmail.com

## License

This project is licensed under the MIT - See the [LICENSE.md](LICENSE.md) file for details

## Built With

* [Pypi](https://pypi.org/)
