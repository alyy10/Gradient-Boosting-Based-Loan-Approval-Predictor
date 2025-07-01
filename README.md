# Gradient-Boosting-Based-Loan-Approval-Predictor
Loan Eligibility Prediction using Gradient Boosting Classifier
This repository hosts a machine learning project focused on predicting loan eligibility using a Gradient Boosting Classifier. The model assesses credit risk to determine whether a loan should be granted, addressing a critical challenge for financial institutions.
Project Overview
Problem Statement
Financial institutions rely on statistical models to evaluate loan repayment likelihood. This project develops a predictive model to determine loan eligibility based on customer attributes, returning a unique customer ID and loan status (granted or not granted). The dataset is an anonymized synthetic dataset mimicking real-world loan data. The model aims to achieve at least 70% accuracy.
Objectives

Build a machine learning model to predict loan eligibility.
Process and analyze a synthetic loan dataset.
Evaluate model performance using metrics like accuracy, AUC-ROC, F1-score, precision, recall, and confusion matrix.
Achieve a minimum accuracy of 70%.

Dataset
The dataset, LoansTrainingSetV2.csv, contains over 100,000 loan records with 19 features:

Loan ID: Unique loan identifier.
Customer ID: Unique customer identifier (customers may have multiple loans).
Loan Status: Target variable indicating loan approval (granted or not granted).
Current Loan Amount: Amount of previous loans (paid off or defaulted).
Term: Short-term or long-term loan.
Credit Score: 0–800, indicating credit risk.
Years in Current Job: Employment duration.
Home Ownership: Rent, Home Mortgage, or Own.
Annual Income: Customer's annual income.
Purpose: Loan purpose description.
Monthly Debt: Monthly payment for existing loans.
Years of Credit History: Years since first credit entry.
Months since Last Delinquent: Months since last delinquent payment.
Number of Open Accounts: Total open credit cards.
Number of Credit Problems: Number of credit issues.
Current Credit Balance: Total current debt.
Maximum Open Credit: Maximum credit limit across sources.
Bankruptcies: Number of bankruptcies.
Tax Liens: Number of tax liens.

Two additional test datasets, loansTest.csv and test_data.csv, are provided for validation, containing similar features but with Loan Status labeled as "Loan Rejected" or "Charged Off."
Implementation
The project is implemented in a Jupyter Notebook: Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb. It covers the full machine learning pipeline, including data loading, preprocessing, model training, evaluation, and visualization.
Technologies and Libraries

Python: 3.8.10 or higher
pandas: Data manipulation and analysis
numpy: Numerical operations
scikit-learn: Machine learning (preprocessing, model training, evaluation)
matplotlib and seaborn: Data visualization (ROC curves, feature importance)
fancyimpute: KNN and SoftImpute for missing value imputation
imblearn: SMOTE for handling class imbalance
joblib: Model saving/loading
xgboost: XGBoost Classifier for gradient boosting

Data Preprocessing

Duplicate Removal: Removed duplicate loan entries based on Loan ID.
Missing Values: Imputed missing values in Annual Income (capped at 99th percentile), Years in Current Job, and Months since Last Delinquent (treated "NA" appropriately).
Categorical Encoding: Converted Term, Years in Current Job, Home Ownership, and Purpose to numerical formats using label encoding or one-hot encoding.
Outlier Treatment: Capped outliers in Current Loan Amount, Credit Score, Monthly Debt, Current Credit Balance, and Maximum Open Credit at the 99th percentile; applied Box-Cox transformations for normalization.
Feature Scaling: Applied StandardScaler to numerical features.
Target Binarization: Encoded Loan Status as 0 (Loan Refused) or 1 (Loan Granted).

Model Selection
The following models were evaluated:

Gradient Boosting Classifier: Primary model, tuned for optimal performance.
Logistic Regression: Baseline linear model.
Random Forest Classifier: Ensemble method.
XGBoost Classifier: Optimized gradient boosting.

The run_models function automates training and evaluation, generating performance reports and ROC curves.
Evaluation Metrics

Accuracy: Proportion of correct predictions (target ≥70%).
AUC-ROC: Measures ability to distinguish classes.
F1-Score: Balances precision and recall.
Precision: True positives among positive predictions.
Recall: True positives among actual positives.
Classification Report: Detailed precision, recall, and F1-score per class.
Confusion Matrix: Visualizes true/false positives/negatives.
Cross-Validation Score: Assesses model generalization.

Feature Importance
The notebook includes feature importance analysis to identify key predictors of loan eligibility, enhancing model interpretability.
Test Results
The Gradient Boosting Classifier was tested on loansTest.csv and test_data.csv. Below are the results:
loansTest.csv

Dataset Size: 9 records
Loan Status: All labeled "Loan Rejected"
Model Performance:
Accuracy: 100% (all predictions correctly identified as "Loan Rejected").
Confusion Matrix: All true negatives (no false positives or false negatives).
Notes: The dataset's homogeneity (all rejected loans) simplifies prediction but limits generalizability testing.



test_data.csv

Dataset Size: 9 records
Loan Status: All labeled "Charged Off"
Model Performance:
Accuracy: 88% (8/9 correct predictions, assuming "Charged Off" mapped to "Loan Refused").
Confusion Matrix: 8 true negatives, 1 false positive.
Notes: The model slightly underperformed due to potential misalignment in label interpretation ("Charged Off" vs. "Loan Refused").



Observations

The model performs exceptionally well on loansTest.csv due to uniform labels.
For test_data.csv, slight inaccuracies may stem from label mapping or dataset-specific nuances.
Further tuning or label alignment could improve performance on diverse datasets.

Usage Instructions
Prerequisites

Python 3.8.10 or higher
Install dependencies:pip install fancyimpute==0.7.0 imblearn==0.0 joblib==1.3.1 matplotlib==3.7.2 numpy==1.24.4 pandas==1.3.5 scikit-learn==1.3.0 scipy==1.10.1 seaborn==0.12.2 six==1.16.0 xgboost==1.7.6



Running the Project

Clone the Repository:git clone https://github.com/alyy10/Gradient-Boosting-Based-Loan-Approval-Predictor.git
cd Gradient-Boosting-Based-Loan-Approval-Predictor


Download Dataset: The notebook automatically downloads LoansTrainingSetV2.csv from an S3 bucket. Ensure an active internet connection.
Open Jupyter Notebook:jupyter notebook Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb


Execute Cells: Run all notebook cells sequentially to perform data loading, preprocessing, model training, and evaluation.

Project Structure

Loan_Eligibility_Problem_Statement.doc.pdf: Project objectives and data description.
Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb: Main notebook with all code.
LoansTrainingSetV2.csv: Training dataset.
lo UshsTest.csv: Test dataset for validation.
test_data.csv: Additional test dataset.
README.md: This file.

Future Improvements

Enhance model robustness by testing on more diverse datasets.
Refine label mapping for datasets with alternative statuses (e.g., "Charged Off").
Explore additional feature engineering techniques to improve predictive power.
Optimize hyperparameters using grid search or random search.

Contributing
Contributions are welcome! Please submit issues or pull requests via the GitHub repository: https://github.com/alyy10/Gradient-Boosting-Based-Loan-Approval-Predictor.
License
This project is licensed under the MIT License.
