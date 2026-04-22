# CREDIT CARD FRAUD DETECTION

This project trains and compares three classifiers on the Kaggle fraud dataset:

- Logistic Regression
- Decision Tree
- Random Forest

Dataset source:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

## 1) Download the dataset from Kaggle

1. Install Kaggle CLI (if not installed):

```powershell
pip install kaggle
```

2. Add your Kaggle API key:

- Create `kaggle.json` from your Kaggle account settings.
- Put it at:
  - Windows: `%USERPROFILE%\.kaggle\kaggle.json`

3. From this repository folder, download and extract:

```powershell
mkdir data
kaggle datasets download -d kartik2112/fraud-detection -p data --unzip
```

After extraction, make sure these files exist:

- `data/fraudTrain.csv`
- `data/fraudTest.csv`

## 2) Run training and evaluation

Default run (uses stratified sampling for speed):

```powershell
python credit_card_fraud_detection.py
```

Use the full dataset:

```powershell
python credit_card_fraud_detection.py --max-train-rows 0 --max-test-rows 0
```

## 3) Output

The script prints:

- Full classification report for each model
- Fraud-class metrics: precision, recall, F1, ROC-AUC, PR-AUC
- Confusion matrix for each model

It also writes a summary CSV:

- `fraud_model_comparison.csv`
