import os
import joblib
import pandas as pd
import numpy as np
from flask import request, jsonify
from google.cloud import storage
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
BUCKET_NAME = "ml-model-data-v1"
MODEL_BLOB_NAME = "rf_chrun_model.pkl"
SCALER_BLOB_NAME = "scaler.pkl"

LOCAL_MODEL_PATH = "/tmp/rf_chrun_model.pkl"
LOCAL_SCALER_PATH = "/tmp/scaler.pkl"

# --- Global Variables ---
model = None
scaler = None
storage_client = storage.Client()

# --- Load Model & Scaler from GCS ---
def _load_model_and_scaler():
    global model, scaler

    try:
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Downloading model from {MODEL_BLOB_NAME}")
            bucket = storage_client.get_bucket(BUCKET_NAME)
            bucket.blob(MODEL_BLOB_NAME).download_to_filename(LOCAL_MODEL_PATH)

        if not os.path.exists(LOCAL_SCALER_PATH):
            logger.info(f"Downloading scaler from {SCALER_BLOB_NAME}")
            bucket = storage_client.get_bucket(BUCKET_NAME)
            bucket.blob(SCALER_BLOB_NAME).download_to_filename(LOCAL_SCALER_PATH)

        model = joblib.load(LOCAL_MODEL_PATH)
        scaler = joblib.load(LOCAL_SCALER_PATH)

        logger.info("Model and Scaler loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load model or scaler: {e}")
        raise RuntimeError("Initialization failed")

try:
    _load_model_and_scaler()
except RuntimeError:
    pass  # Cloud Function warm-up will retry

# --- Feature Validation ---
def feature_validation(json_data):
    required_features = {
        'CustomerID', 'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
        'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'
    }
    missing = set()
    for i, data in enumerate(json_data):
        missing.update(required_features - set(data.keys()))
    return list(missing)

# --- Prediction Function ---
def predict(json_data):
    df = pd.DataFrame(json_data)
    CustomerID_lst = df["CustomerID"].to_list()
    df.drop("CustomerID", axis=1, inplace=True)

    df["Gender"].replace({"Male": 1, "Female": 0}, inplace=True)
    df["Subscription Type"].replace({"Basic": 0, "Standard": 1, "Premium": 2}, inplace=True)
    df["Contract Length"].replace({"Monthly": 0, "Quarterly": 1, "Annual": 2}, inplace=True)

    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)

    return [{"CustomerID": CustomerID_lst[i], "prediction": int(prediction[i])} for i in range(len(CustomerID_lst))]

# --- Cloud Function Entry Point ---
def predict_churn(request):
    headers = {'Access-Control-Allow-Origin': '*'}

    if request.method == 'OPTIONS':
        return ('', 204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        })

    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500, headers

    try:
        json_data = request.get_json(silent=True)
        if not json_data:
            return jsonify({"error": "Invalid JSON"}), 400, headers

        missing = feature_validation(json_data)
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400, headers

        predictions = predict(json_data)
        return jsonify(predictions), 200, headers

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500, headers
