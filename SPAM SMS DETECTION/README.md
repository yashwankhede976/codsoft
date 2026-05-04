# SMS Spam Detection (Kaggle Dataset)

This project trains an AI model to classify SMS messages as **spam** or **legitimate (ham)** using:

- **TF-IDF** text features
- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machine (Linear SVM)**

Dataset source: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Train and compare models

```bash
python train_spam_classifier.py
```

What this does:
- Downloads dataset from Kaggle (`uciml/sms-spam-collection-dataset`) using `kagglehub`
- Preprocesses labels/messages
- Trains all three models on a stratified train/test split
- Compares metrics (accuracy, precision, recall, F1)
- Saves the best model and metrics

Artifacts created in `artifacts/`:
- `best_sms_spam_model.joblib`
- `metrics.json`

Optional arguments:

```bash
python train_spam_classifier.py --test-size 0.2 --random-state 42 --max-features 20000
```

If you already have `spam.csv` locally:

```bash
python train_spam_classifier.py --dataset-csv "path/to/spam.csv"
```

## 3) Predict new SMS text

```bash
python predict_sms.py "Congratulations! You've won a free iPhone. Click now!"
```

Example output:

```text
Message: Congratulations! You've won a free iPhone. Click now!
Prediction: spam
```
