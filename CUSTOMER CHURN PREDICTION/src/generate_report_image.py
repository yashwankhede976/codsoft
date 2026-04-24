from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def get_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    input_csv = project_root / "reports" / "model_comparison.csv"
    output_image = project_root / "reports" / "model_comparison.png"
    return input_csv, output_image


def generate_image(input_csv: Path, output_image: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Report CSV not found at: {input_csv}. Run train_churn_model.py first."
        )

    df = pd.read_csv(input_csv)
    required_cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    table_df = df[required_cols].copy()
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        table_df[col] = table_df[col].map(lambda value: f"{value:.4f}")

    fig_height = 1.6 + 0.55 * len(table_df)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values.tolist(),
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f2937")
        else:
            cell.set_facecolor("#f9fafb" if row % 2 else "#eef2ff")

    ax.set_title("Customer Churn Model Comparison", fontsize=14, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_image, dpi=220, bbox_inches="tight")


def main() -> None:
    input_csv, output_image = get_paths()
    generate_image(input_csv, output_image)
    print(f"Saved image: {output_image}")


if __name__ == "__main__":
    main()
