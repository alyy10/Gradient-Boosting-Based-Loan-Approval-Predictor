Loan Eligibility Prediction using Gradient Boosting Classifier
A machine learning project to predict loan eligibility using a Gradient Boosting Classifier, addressing credit risk assessment for financial institutions.
Table of Contents

Project Overview
Problem Statement
Objectives


Dataset
Training Dataset
Test Datasets


Implementation
Technologies and Libraries
Data Preprocessing
Model Selection
Evaluation Metrics
Feature Importance


Test Results
loansTest.csv
test_data.csv


Usage Instructions
Prerequisites
Running the Project


Project Structure
Future Improvements
Contributing
License

Project Overview
Problem Statement
Financial institutions use statistical models to assess loan repayment likelihood. This project develops a predictive model to determine loan eligibility based on customer attributes, returning a unique customer ID and loan status (granted or not granted). The dataset is synthetic, mimicking real-world loan data. The target is to achieve at least 70% accuracy.
Objectives

Develop a machine learning model for loan eligibility prediction.
Process and analyze a synthetic loan dataset.
Evaluate model performance using accuracy, AUC-ROC, F1-score, precision, recall, and confusion matrix.
Ensure model accuracy meets or exceeds 70%.

Dataset
Training Dataset

File: LoansTrainingSetV2.csv
Size: Over 100,000 loan records
Features (19):
Loan ID: Unique loan identifier
Customer ID: Unique customer identifier (multiple loans possible per customer)
Loan Status: Target variable (granted or not granted)
Current Loan Amount: Amount of prior loans (paid off or defaulted)
Term: Short-term or long-term
Credit Score: 0–800, indicating credit risk
Years in Current Job: Employment duration
Home Ownership: Rent, Home Mortgage, or Own
Annual Income: Customer's annual income
Purpose: Loan purpose
Monthly Debt: Monthly payment for existing loans
Years of Credit History: Years since first credit entry
Months since Last Delinquent: Months since last delinquent payment
Number of Open Accounts: Open credit cards
Number of Credit Problems: Credit issues
Current Credit Balance: Total current debt
Maximum Open Credit: Maximum credit limit
Bankruptcies: Number of bankruptcies
Tax Liens: Number of tax liens



Test Datasets

loansTest.csv: 9 records, all labeled "Loan Rejected"
test_data.csv: 9 records, all labeled "Charged Off"

Implementation
Technologies and Libraries

Python: 3.8.10+
pandas: Data manipulation
numpy: Numerical operations
scikit-learn: Machine learning (preprocessing, training, evaluation)
matplotlib, seaborn: Visualization (ROC curves, feature importance)
fancyimpute: KNN and SoftImpute for missing values
imblearn: SMOTE for class imbalance
joblib: Model persistence
xgboost: XGBoost Classifier

Data Preprocessing

Duplicate Removal: Removed duplicates using Loan ID.
Missing Values: Imputed Annual Income (capped at 99th percentile), Years in Current Job, and Months since Last Delinquent (handled "NA").
Categorical Encoding: Converted Term, Years in Current Job, Home Ownership, and Purpose to numerical formats (label or one-hot encoding).
Outlier Treatment: Capped outliers in Current Loan Amount, Credit Score, Monthly Debt, Current Credit Balance, and Maximum Open Credit at 99th percentile; applied Box-Cox transformations.
Feature Scaling: Used StandardScaler for numerical features.
Target Binarization: Encoded Loan Status as 0 (Refused) or 1 (Granted).

Model Selection

Primary Model: Gradient Boosting Classifier (tuned)
Other Models:
Logistic Regression (baseline)
Random Forest Classifier
XGBoost Classifier


Automation: run_models function for training and evaluation with performance reports and ROC curves.

Evaluation Metrics

Accuracy: Correct predictions (target ≥70%)
AUC-ROC: Class discrimination ability
F1-Score: Precision-recall balance
Precision: True positives among positive predictions
Recall: True positives among actual positives
Classification Report: Per-class precision, recall, F1-score
Confusion Matrix: True/false positives/negatives
Cross-Validation Score: Model generalization

Feature Importance
Analyzed to identify key predictors, enhancing model interpretability.
Test Results
loansTest.csv

Size: 9 records
Loan Status: All "Loan Rejected"
Performance:
Accuracy: 100% (all correctly predicted)
Confusion Matrix: All true negatives
Notes: Homogeneous labels simplify prediction but limit generalizability.



test_data.csv

Size: 9 records
Loan Status: All "Charged Off"
Performance:
Accuracy: 88% (8/9 correct, assuming "Charged Off" = "Loan Refused")
Confusion Matrix: 8 true negatives, 1 false positive
Notes: Slight inaccuracies due to potential label mapping issues.



Usage Instructions
Prerequisites

Python 3.8.10+
Install dependencies:pip install fancyimpute==0.7.0 imblearn==0.0 joblib==1.3.1 matplotlib==3.7.2 numpy==1.24.4 pandas==1.3.5 scikit-learn==1.3.0 scipy==1.10.1 seaborn==0.12.2 six==1.16.0 xgboost==1.7.6



Running the Project

Clone Repository:git clone https://github.com/alyy10/Gradient-Boosting-Based-Loan-Approval-Predictor.git
cd Gradient-Boosting-Based-Loan-Approval-Predictor


Dataset Access: LoansTrainingSetV2.csv is downloaded automatically via S3 (requires internet).
Run Notebook:jupyter notebook Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb


Execute Cells: Run all cells sequentially for data processing, training, and evaluation.

Project Structure

Loan_Eligibility_Problem_Statement.doc.pdf: Project objectives and data details
Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb: Core notebook
LoansTrainingSetV2.csv: Training dataset
loansTest.csv: Test dataset
test_data.csv: Additional test dataset
README.md: This file

Future Improvements

Test on diverse datasets for robustness.
Refine label mapping for alternative statuses (e.g., "Charged Off").
Explore advanced feature engineering.
Optimize hyperparameters via grid or random search.

Contributing
Submit issues or pull requests at: https://github.com/alyy10/Gradient-Boosting-Based-Loan-Approval-Predictor.
License
MIT License
