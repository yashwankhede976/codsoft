"""
Create a PNG output image from fraud_model_comparison.csv.

Run:
  python fraud_output_visualizer.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "fraud_model_comparison.csv"
    out_path = base_dir / "fraud_output_summary.png"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run credit_card_fraud_detection.py first."
        )

    df = pd.read_csv(csv_path).sort_values("f1_fraud", ascending=False).reset_index(drop=True)
    models = df["model"].tolist()

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    x = list(range(len(models)))
    width = 0.36

    ax1.bar([i - width / 2 for i in x], df["f1_fraud"], width=width, color="#2563eb", label="F1 (Fraud)")
    ax1.bar([i + width / 2 for i in x], df["recall_fraud"], width=width, color="#0ea5e9", label="Recall (Fraud)")
    ax1.set_title("Fraud-Class Performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    ax2.bar([i - width / 2 for i in x], df["roc_auc"], width=width, color="#16a34a", label="ROC-AUC")
    ax2.bar([i + width / 2 for i in x], df["pr_auc"], width=width, color="#f59e0b", label="PR-AUC")
    ax2.set_title("Ranking Metrics")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()

    ax3.axis("off")
    view = df[["model", "precision_fraud", "recall_fraud", "f1_fraud", "roc_auc", "pr_auc"]].copy()
    for col in view.columns[1:]:
        view[col] = view[col].map(lambda v: f"{v:.4f}")
    table = ax3.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    ax3.set_title("Model Comparison Table", pad=10)

    fig.suptitle("Credit Card Fraud Detection Output", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Image saved: {out_path}")


if __name__ == "__main__":
    main()
