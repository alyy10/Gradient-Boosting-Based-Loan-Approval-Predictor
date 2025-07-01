# Loan Eligibility Prediction using Gradient Boosting Classifier

A machine learning project to predict loan eligibility using a Gradient Boosting Classifier, addressing credit risk assessment for financial institutions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement & Objectives](#problem-statement--objectives)
- [Dataset](#dataset)
  - [Training Dataset](#training-dataset)
  - [Test Datasets](#test-datasets)
- [Implementation](#implementation)
  - [Technologies and Libraries](#technologies-and-libraries)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Selection](#model-selection)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Feature Importance](#feature-importance)
- [Test Results](#test-results)
- [Usage Instructions](#usage-instructions)
  - [Prerequisites](#prerequisites)
  - [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project builds a machine learning model to predict loan eligibility based on customer attributes. It uses a synthetic dataset that simulates real-world loan data.

---

## Problem Statement & Objectives

Financial institutions rely on statistical models to estimate loan repayment likelihood.

Objectives:
- Develop a predictive model returning customer ID and loan status (granted or not granted)
- Achieve at least 70% accuracy
- Evaluate performance using metrics like AUC-ROC, F1-score, precision, recall, and confusion matrix

---

## Dataset

### Training Dataset

- File: `LoansTrainingSetV2.csv`
- Records: Over 100,000
- Features (19):
  - Loan ID, Customer ID, Loan Status, Current Loan Amount, Term, Credit Score, Years in Current Job, Home Ownership, Annual Income, Purpose, Monthly Debt, Years of Credit History, Months since Last Delinquent, Number of Open Accounts, Number of Credit Problems, Current Credit Balance, Maximum Open Credit, Bankruptcies, Tax Liens

### Test Datasets

- `loansTest.csv`: 9 records, all labeled "Loan Rejected"
- `test_data.csv`: 9 records, all labeled "Charged Off"

---

## Implementation

### Technologies and Libraries

- Python 3.8.10+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- fancyimpute (KNN, SoftImpute)
- imblearn (SMOTE)
- joblib
- xgboost

### Data Preprocessing

- Removed duplicates by Loan ID
- Imputed missing values (Annual Income, Years in Current Job, Months since Last Delinquent)
- Label/one-hot encoding for categorical features
- Outlier treatment (capped at 99th percentile, Box-Cox transformations)
- Feature scaling with StandardScaler
- Target encoding: Loan Status → 0 (Refused) / 1 (Granted)

### Model Selection

- Primary: Gradient Boosting Classifier (tuned)
- Other models: Logistic Regression, Random Forest Classifier, XGBoost Classifier
- Automation: `run_models` function for training, evaluation, and ROC curves

### Evaluation Metrics

- Accuracy (≥70%)
- AUC-ROC
- F1-Score
- Precision
- Recall
- Confusion Matrix
- Cross-validation score

### Feature Importance

Identifies key predictors to explain the model.

---

## Test Results

| Dataset        | Records | Accuracy | Notes                                       |
|----------------|--------:|--------:|----------------------------------------------|
| loansTest.csv  |       9 |   100%   | All correctly predicted (homogeneous labels) |
| test_data.csv  |       9 |    88%   | 8/9 correct; slight label mapping issue      |

---

## Usage Instructions

### Prerequisites

Python 3.8.10+  
Install dependencies:

```bash
pip install fancyimpute==0.7.0 imblearn==0.0 joblib==1.3.1 matplotlib==3.7.2 numpy==1.24.4 pandas==1.3.5 scikit-learn==1.3.0 scipy==1.10.1 seaborn==0.12.2 six==1.16.0 xgboost==1.7.6

### Running the Project

# Clone the repository
git clone https://github.com/alyy10/Gradient-Boosting-Based-Loan-Approval-Predictor.git
cd Gradient-Boosting-Based-Loan-Approval-Predictor

# Dataset is auto-downloaded via S3 (requires internet)
# Run notebook
jupyter notebook Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb


### Project Structure

├── Loan_Eligibility_Problem_Statement.doc.pdf
├── Loan_Eligibility_Prediction_using_Gradient_Boosting_Classifier.ipynb
├── LoansTrainingSetV2.csv
├── loansTest.csv
├── test_data.csv
└── README.md

