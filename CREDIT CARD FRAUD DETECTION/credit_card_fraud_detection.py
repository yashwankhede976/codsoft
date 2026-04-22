"""
Credit Card Fraud Detection
Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection

Expected files:
  data/fraudTrain.csv
  data/fraudTest.csv

Run:
  python credit_card_fraud_detection.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fraud detection models on the Kaggle card transaction dataset."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/fraudTrain.csv"),
        help="Path to fraudTrain.csv",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/fraudTest.csv"),
        help="Path to fraudTest.csv",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=300_000,
        help=(
            "Maximum number of training rows to use. "
            "Set to 0 to use all rows."
        ),
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=150_000,
        help=(
            "Maximum number of test rows to use. "
            "Set to 0 to use all rows."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("fraud_model_comparison.csv"),
        help="Where to save the model comparison table as CSV.",
    )
    return parser.parse_args()


def stratified_sample(
    df: pd.DataFrame, target_col: str, max_rows: int, random_state: int
) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    ratio = max_rows / len(df)
    sampled = df.groupby(target_col, group_keys=False).sample(
        frac=ratio, random_state=random_state
    )
    sampled = sampled.reset_index(drop=True)
    return sampled


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}\n"
            "Download from Kaggle and place fraudTrain.csv/fraudTest.csv in the data folder."
        )
    return pd.read_csv(path)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "trans_date_trans_time" in out.columns:
        dt = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
        out["trans_hour"] = dt.dt.hour
        out["trans_day"] = dt.dt.day
        out["trans_month"] = dt.dt.month
        out["trans_weekday"] = dt.dt.weekday
        out["is_weekend"] = (out["trans_weekday"] >= 5).astype(int)

    if "dob" in out.columns and "trans_date_trans_time" in out.columns:
        dob = pd.to_datetime(out["dob"], errors="coerce")
        trans_dt = pd.to_datetime(out["trans_date_trans_time"], errors="coerce")
        age_years = (trans_dt - dob).dt.days / 365.25
        out["customer_age"] = age_years.clip(lower=0, upper=120)

    return out


def build_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "is_fraud" not in df.columns:
        raise ValueError("Target column 'is_fraud' not found in dataframe.")

    df = add_time_features(df)
    y = df["is_fraud"].astype(int)

    drop_cols = [
        "is_fraud",
        "Unnamed: 0",
        "trans_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "dob",
        "trans_date_trans_time",
    ]
    available_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=available_drop_cols)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    preprocessor = build_preprocessor(X_train)

    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=16,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
    }

    rows = []
    print("\nModel Evaluation")
    print("=" * 80)

    for name, estimator in models.items():
        model = Pipeline(
            steps=[("preprocess", clone(preprocessor)), ("model", estimator)]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        print(f"\n{name}")
        print("-" * 80)
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Legitimate (0)", "Fraud (1)"],
                zero_division=0,
            )
        )
        print(f"Confusion Matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
        print(
            "Fraud class metrics -> "
            f"Precision:{precision:.4f} Recall:{recall:.4f} "
            f"F1:{f1:.4f} ROC-AUC:{roc_auc:.4f} PR-AUC:{pr_auc:.4f}"
        )

        rows.append(
            {
                "model": name,
                "precision_fraud": precision,
                "recall_fraud": recall,
                "f1_fraud": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["f1_fraud", "pr_auc", "recall_fraud"], ascending=False
    )
    return results


def main() -> None:
    args = parse_args()

    print("Credit Card Fraud Detection")
    print("Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection")

    train_df = load_dataset(args.train_path)
    test_df = load_dataset(args.test_path)

    train_df = stratified_sample(
        train_df, "is_fraud", args.max_train_rows, args.random_state
    )
    test_df = stratified_sample(
        test_df, "is_fraud", args.max_test_rows, args.random_state
    )

    X_train, y_train = build_features_and_target(train_df)
    X_test, y_test = build_features_and_target(test_df)

    print(f"\nTrain rows: {len(X_train):,} | Fraud rate: {y_train.mean():.4%}")
    print(f"Test rows:  {len(X_test):,} | Fraud rate: {y_test.mean():.4%}")

    results = evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=args.random_state,
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_path, index=False)

    print("\nModel Comparison (sorted by fraud F1)")
    print("=" * 80)
    print(
        results[
            [
                "model",
                "precision_fraud",
                "recall_fraud",
                "f1_fraud",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved comparison table to: {args.results_path.resolve()}")


if __name__ == "__main__":
    main()
