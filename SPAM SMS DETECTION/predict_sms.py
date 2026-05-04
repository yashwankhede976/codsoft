from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether an SMS message is spam or ham.")
    parser.add_argument(
        "message",
        type=str,
        help="SMS text to classify.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts") / "best_sms_spam_model.joblib",
        help="Path to trained model pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Train first with `python train_spam_classifier.py`."
        )

    model = joblib.load(args.model_path)
    prediction = model.predict([args.message])[0]

    print(f"Message: {args.message}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
