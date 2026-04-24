from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def get_paths() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "Churn_Modelling.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    return data_path, models_dir, reports_dir


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    return pd.read_csv(data_path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = df.copy()
    target_col = "Exited"
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    feature_df = df.drop(columns=drop_cols + [target_col])
    target = df[target_col]

    categorical_cols = ["Geography", "Gender"]
    numeric_cols = [col for col in feature_df.columns if col not in categorical_cols]
    return feature_df, target, categorical_cols, numeric_cols


def build_models(categorical_cols: list[str], numeric_cols: list[str]) -> dict[str, Pipeline]:
    onehot = OneHotEncoder(handle_unknown="ignore")

    lr_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", onehot, categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", onehot, categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", lr_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, Pipeline]]:
    scores: list[dict[str, float | str]] = []
    diagnostics: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        scores.append(metrics)

        diagnostics[name] = {
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        fitted_models[name] = model

    results_df = pd.DataFrame(scores).sort_values(by="roc_auc", ascending=False)
    return results_df, diagnostics, fitted_models


def save_outputs(
    results_df: pd.DataFrame,
    diagnostics: dict[str, dict[str, object]],
    fitted_models: dict[str, Pipeline],
    models_dir: Path,
    reports_dir: Path,
) -> tuple[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = reports_dir / "model_comparison.csv"
    diagnostics_path = reports_dir / "classification_diagnostics.json"
    summary_path = reports_dir / "summary.txt"

    results_df.to_csv(comparison_path, index=False)
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    best_model_name = str(results_df.iloc[0]["model"])
    best_model_path = models_dir / "best_churn_model.joblib"
    joblib.dump(fitted_models[best_model_name], best_model_path)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Customer Churn Prediction - Model Summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"Best model: {best_model_name}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")

    return best_model_name, best_model_path


def main() -> None:
    data_path, models_dir, reports_dir = get_paths()
    df = load_data(data_path)
    x, y, categorical_cols, numeric_cols = prepare_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_models(categorical_cols, numeric_cols)
    results_df, diagnostics, fitted_models = evaluate_models(
        models, x_train, x_test, y_train, y_test
    )
    best_model_name, best_model_path = save_outputs(
        results_df, diagnostics, fitted_models, models_dir, reports_dir
    )

    print("Training complete.")
    print(f"Dataset rows: {len(df)}")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model path: {best_model_path}")


if __name__ == "__main__":
    main()
