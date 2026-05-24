import io
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from src.features.feature_engineering import align_features

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "churn_xgb_model.pkl"

app = Flask(__name__)
app.secret_key = "replace-me-with-a-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def load_model():
    return joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("Please choose a CSV file to upload.")
        return redirect(url_for("index"))

    try:
        input_df = pd.read_csv(file)
    except Exception as exc:
        flash(f"Unable to read CSV file: {exc}")
        return redirect(url_for("index"))

    if "user_id" not in input_df.columns:
        flash("CSV must include a 'user_id' column.")
        return redirect(url_for("index"))

    model = load_model()
    features = align_features(input_df)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    output_df = input_df.copy()
    for col in features.columns:
        output_df[col] = features[col].values
    output_df["predicted_churn"] = predictions.astype(int)
    output_df["predicted_probability"] = probabilities.round(4)

    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="churn_predictions.csv",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
