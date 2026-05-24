import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_churn_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_xgb_model.pkl"
REPORT_PATH = ROOT_DIR / "reports" / "evaluation_report.txt"
PREDICTIONS_PATH = ROOT_DIR / "data" / "processed" / "evaluation_predictions.csv"


def evaluate_model() -> None:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    X = df.drop(columns=["user_id", "churned"])
    y = df["churned"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    report = classification_report(y, y_pred, digits=4)
    confusion = confusion_matrix(y, y_pred)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        fp.write("Classification report:\n")
        fp.write(report)
        fp.write("\nConfusion matrix:\n")
        fp.write(str(confusion))

    predictions = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "churned": y,
            "predicted_churn": y_pred,
            "predicted_probability": y_proba,
        }
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Evaluation report saved to {REPORT_PATH}")
    print(f"Prediction file saved to {PREDICTIONS_PATH}")
    print(report)


if __name__ == "__main__":
    evaluate_model()
