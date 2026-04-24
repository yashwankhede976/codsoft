# CUSTOMER CHURN PREDICTION

This project predicts customer churn for a subscription-style business using the Kaggle dataset:

- https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction

## What this project does

- Downloads and uses `Churn_Modelling.csv`.
- Builds a preprocessing pipeline for demographic and behavior features.
- Trains and compares:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Evaluates models using `accuracy`, `precision`, `recall`, `f1_score`, and `roc_auc`.
- Saves the best model and reports.

## Project structure

```text
CUSTOMER CHURN PREDICTION/
|-- data/
|   |-- Churn_Modelling.csv
|-- models/
|   |-- best_churn_model.joblib
|-- reports/
|   |-- model_comparison.csv
|   |-- classification_diagnostics.json
|   |-- summary.txt
|-- src/
|   |-- train_churn_model.py
|-- requirements.txt
|-- README.md
```

## Run

From this folder:

```powershell
C:\Users\umesh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src\train_churn_model.py
```

## Kaggle download command used

```powershell
kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction -p data
```

## Current results (test split)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8710 | 0.7876 | 0.5012 | 0.6126 | 0.8704 |
| Random Forest | 0.8595 | 0.7669 | 0.4447 | 0.5630 | 0.8543 |
| Logistic Regression | 0.7135 | 0.3872 | 0.7002 | 0.4987 | 0.7772 |

Best saved model: `models/best_churn_model.joblib` (Gradient Boosting)

## Notes on features

- Target column: `Exited` (1 = churn, 0 = not churned)
- Dropped identifiers: `RowNumber`, `CustomerId`, `Surname`
- Categorical features: `Geography`, `Gender`
- Numeric features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
