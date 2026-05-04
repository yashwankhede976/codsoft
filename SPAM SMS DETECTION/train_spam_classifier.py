from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATASET_SLUG = "uciml/sms-spam-collection-dataset"


def download_kaggle_dataset() -> Path:
    """Download the Kaggle SMS Spam dataset and return the CSV path."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is required to download the dataset. Install it with: pip install kagglehub"
        ) from exc

    dataset_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset directory: {dataset_dir}")

    return csv_files[0]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load dataset and normalize to ['label', 'message'] columns."""
    df = pd.read_csv(csv_path, encoding="latin-1")

    if {"v1", "v2"}.issubset(df.columns):
        normalized = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif {"label", "message"}.issubset(df.columns):
        normalized = df[["label", "message"]].copy()
    else:
        normalized = df.iloc[:, :2].copy()
        normalized.columns = ["label", "message"]

    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    normalized["message"] = normalized["message"].astype(str).str.strip()
    normalized = normalized[normalized["label"].isin(["ham", "spam"])].dropna()

    if normalized.empty:
        raise ValueError("Loaded dataset is empty after preprocessing.")

    return normalized


def build_models(max_features: int | None = None) -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )

    return {
        "naive_bayes": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", MultinomialNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
    }
    return metrics


def train_and_compare(df: pd.DataFrame, test_size: float, random_state: int, max_features: int | None) -> dict[str, Any]:
    x = df["message"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = build_models(max_features=max_features)

    leaderboard: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        leaderboard[name] = metrics
        trained_models[name] = model

    best_name = max(leaderboard, key=lambda model_name: leaderboard[model_name]["f1"])

    return {
        "best_model_name": best_name,
        "best_model": trained_models[best_name],
        "leaderboard": leaderboard,
        "dataset_size": int(len(df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMS spam classifier on Kaggle SMS Spam Collection dataset.")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=None,
        help="Optional path to a local dataset CSV. If omitted, dataset is downloaded via kagglehub.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for test split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max-features", type=int, default=None, help="Optional cap for TF-IDF vocabulary size.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to store trained model and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_csv:
        csv_path = args.dataset_csv
    else:
        csv_path = download_kaggle_dataset()

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = load_dataset(csv_path)
    result = train_and_compare(
        df=df,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best_sms_spam_model.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(result["best_model"], model_path)

    serializable = {
        "dataset_slug": DATASET_SLUG,
        "dataset_csv": str(csv_path),
        "dataset_size": result["dataset_size"],
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "best_model_name": result["best_model_name"],
        "leaderboard": result["leaderboard"],
        "model_path": str(model_path),
    }

    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Dataset: {csv_path}")
    print(f"Best model: {result['best_model_name']}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nLeaderboard (higher F1 is better):")

    for model_name, scores in sorted(
        result["leaderboard"].items(), key=lambda item: item[1]["f1"], reverse=True
    ):
        print(
            f"- {model_name}: "
            f"accuracy={scores['accuracy']:.4f}, "
            f"precision={scores['precision']:.4f}, "
            f"recall={scores['recall']:.4f}, "
            f"f1={scores['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
