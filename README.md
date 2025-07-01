# Gradient-Boosting-Based-Loan-Approval-Predictor
Loan Eligibility Prediction using Gradient Boosting Classifier

This repository contains a machine learning project aimed at predicting loan eligibility based on various customer attributes. The project utilizes a Gradient Boosting Classifier to determine whether a loan should be granted to an individual, addressing a common challenge faced by financial institutions in assessing credit risk.

Problem Statement

When customers apply for loans, financial institutions use statistical models to assess the likelihood of repayment and determine whether to grant the loan. This project addresses the complexity of this assessment by developing a predictive model. The primary objective is to implement a machine learning model that predicts loan eligibility based on provided customer data. The model should return a unique customer ID and a loan status label (granted or not granted). The dataset used is an anonymized synthetic dataset designed to mimic real-world loan data characteristics. A key evaluation criterion for the model is to achieve an accuracy of at least 70%.

Dataset

The dataset, LoansTrainingSetV2.csv, consists of over 100,000 loan records. It contains 19 features, including unique identifiers for loans and customers, loan status, current loan amount, term, credit score, employment history, home ownership, annual income, loan purpose, monthly debt, years of credit history, months since last delinquent payment, number of open accounts, number of credit problems, current credit balance, maximum open credit, bankruptcies, and tax liens.

Key Features:

•
Loan ID: A unique identifier for the loan information.

•
Customer ID: A unique identifier for the customer. Customers may have more than one loan.

•
Loan Status: A categorical variable indicating if the loan was given to this customer (Target Variable).

•
Current Loan Amount: The loan amount that was either completely paid off or defaulted. This data pertains to previous loans.

•
Term: A categorical variable indicating if it is a short-term or long-term loan.

•
Credit Score: A value between 0 and 800 indicating the riskiness of the borrower’s credit history.

•
Years in current job: A categorical variable indicating how many years the customer has been in their current job.

•
Home Ownership: Categorical variable indicating home ownership. Values include "Rent", "Home Mortgage", and "Own". If the value is "Own", the customer is a homeowner with no mortgage.

•
Annual Income: The customer's annual income.

•
Purpose: A description of the purpose of the loan.

•
Monthly Debt: The customer's monthly payment for their existing loans.

•
Years of Credit History: The years since the first entry in the customer’s credit history.

•
Months since last delinquent: Months since the last loan delinquent payment.

•
Number of Open Accounts: The total number of open credit cards.

•
Number of Credit Problems: The number of credit problems in the customer records.

•
Current Credit Balance: The current total debt for the customer.

•
Maximum Open Credit: The maximum credit limit for all credit sources.

•
Bankruptcies: The number of bankruptcies.

•
Tax Liens: The number of tax liens.

Implementation Details

The core of this project is implemented in a Jupyter Notebook titled Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb. The notebook details the entire machine learning pipeline, from data loading and preprocessing to model training, evaluation, and selection.

Technologies and Libraries

The project leverages several popular Python libraries for data manipulation, machine learning, and visualization:

•
pandas: For data loading, manipulation, and analysis.

•
numpy: For numerical operations.

•
scikit-learn: A comprehensive library for machine learning, used for data splitting, preprocessing (e.g., LabelBinarizer, StandardScaler, OrdinalEncoder), and various classification models (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, etc.).

•
matplotlib and seaborn: For data visualization, including ROC curves and feature importance plots.

•
fancyimpute: Specifically KNN and SoftImpute, for handling missing values.

•
imblearn: For addressing class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

•
joblib: For saving and loading trained machine learning models.

•
xgboost: For the XGBoost Classifier, a highly efficient and flexible gradient boosting library.

Data Preprocessing and Feature Engineering

The notebook performs several crucial data preprocessing steps to prepare the raw data for model training:

1.
Duplicate Removal: Duplicate loan entries are identified and removed based on the 'Loan ID' to ensure data integrity.

2.
Handling Missing Values: Missing values in various columns are addressed. For instance, 'Annual Income' outliers are capped at the 99th percentile, and missing income values are imputed. Categorical features like 'Years in current job' and 'Months since last delinquent' are also handled, with 'NA' values being imputed or treated appropriately.

3.
Categorical Feature Encoding: Categorical variables such as 'Term', 'Years in current job', 'Home Ownership', and 'Purpose' are converted into numerical representations suitable for machine learning algorithms. This includes mapping textual categories to numerical labels and one-hot encoding where appropriate.

4.
Outlier Treatment: Outliers in numerical features like 'Current Loan Amount', 'Credit Score', 'Monthly Debt', 'Current Credit Balance', and 'Maximum Open Credit' are identified and treated, often by capping values at certain percentiles (e.g., 99th percentile) or using transformations like Box-Cox to normalize distributions.

5.
Feature Scaling: Numerical features are scaled using StandardScaler to ensure that all features contribute equally to the model training process, preventing features with larger values from dominating the learning.

6.
Target Variable Binarization: The 'Loan Status' target variable is binarized into a numerical format (e.g., 0 for 'Loan Refused' and 1 for 'Loan Granted').

Model Selection and Training

The project explores multiple classification algorithms to identify the most effective model for loan eligibility prediction. The models evaluated include:

•
Gradient Boosting Classifier: The primary model, tuned for optimal performance.

•
Logistic Regression: A baseline linear model.

•
Random Forest Classifier: An ensemble learning method.

•
XGBoost Classifier: An optimized distributed gradient boosting library.

The notebook includes functions to train these models, evaluate their performance using various metrics, and compare their effectiveness. The run_models function is designed to automate the training and evaluation process across different classifiers, generating performance reports and ROC curves.

Evaluation Metrics

Model performance is assessed using a range of metrics critical for classification tasks, especially in imbalanced datasets:

•
Accuracy: The proportion of correctly classified instances.

•
Area Under the Receiver Operating Characteristic (ROC) Curve (AUC-ROC): A measure of the model's ability to distinguish between classes, particularly useful for imbalanced datasets.

•
F1-Score: The harmonic mean of precision and recall, providing a balance between the two.

•
Precision: The proportion of true positive predictions among all positive predictions.

•
Recall: The proportion of true positive predictions among all actual positive instances.

•
Classification Report: Provides a detailed breakdown of precision, recall, and F1-score for each class.

•
Confusion Matrix: Visualizes the performance of a classification algorithm, showing true positives, true negatives, false positives, and false negatives.

•
Cross-Validation Score: Used to assess the model's generalization ability and robustness.

Feature Importance

The notebook also includes functionality to determine feature importance, helping to understand which input variables have the most significant impact on the loan eligibility prediction. This is crucial for interpretability and potentially for feature selection in future iterations of the model.

Usage

To run this project and reproduce the results, follow these steps:

Prerequisites

Ensure you have Python 3.8.10 or higher installed. The following Python libraries are required:

•
fancyimpute==0.7.0

•
imblearn==0.0

•
joblib==1.3.1

•
matplotlib==3.7.2

•
numpy==1.24.4

•
pandas==1.3.5

•
scikit-learn==1.3.0

•
scipy==1.10.1

•
seaborn==0.12.2

•
six==1.16.0

•
xgboost==1.7.6

You can install these dependencies using pip:

Bash


pip install fancyimpute==0.7.0 imblearn==0.0 joblib==1.3.1 matplotlib==3.7.2 numpy==1.24.4 pandas==1.3.5 scikit-learn==1.3.0 scipy==1.10.1 seaborn==0.12.2 six==1.16.0 xgboost==1.7.6


Running the Notebook

1.
Clone the repository
2.
Download the dataset: The notebook directly downloads the LoansTrainingSetV2.csv from an S3 bucket. Ensure you have an active internet connection.

3.
Open the Jupyter Notebook:

4.
Execute cells: Run all the cells in the notebook sequentially. This will perform data loading, preprocessing, model training, and evaluation.

Project Structure

•
Loan_Eligibility_Problem_Statement.doc.pdf: Original problem statement detailing the project's objectives and data description.

•
Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb: The main Jupyter Notebook containing all the code for data analysis, model development, and evaluation.

•
LoansTrainingSetV2.csv: The dataset used for training and testing the models.

•
README.md: This file, providing an overview of the project.

