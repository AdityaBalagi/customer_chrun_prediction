
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Load Model 
model = joblib.load("rf_chrun_model.pkl")
# Load Scaler
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict(json_data):
    df = pd.DataFrame(json_data)
    CustomerID_lst = df["CustomerID"].to_list()
    df.drop("CustomerID", axis=1, inplace=True)

    # Encode categorical values
    df["Gender"].replace({"Male": 1, "Female": 0}, inplace=True)
    df["Subscription Type"].replace({"Basic": 0, "Standard": 1, "Premium": 2}, inplace=True)
    df["Contract Length"].replace({"Monthly": 0, "Quarterly": 1, "Annual": 2}, inplace=True)

    # Normalize data with MinMaxScaler
    df_scaled = scaler.transform(df)  # WARNING: Should ideally use the same scaler used during training

    # Make prediction
    prediction = model.predict(df_scaled)

    result = []
    for i in range(len(CustomerID_lst)):
        output = {"CustomerID": CustomerID_lst[i], "prediction": int(prediction[i])}
        result.append(output)

    return result

def feature_validation(json_data):
    required_features = {'CustomerID', 'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
                         'Payment Delay', 'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction'}
    missing = set()
    for i, data in enumerate(json_data):
        missing.update(required_features - set(data.keys()))
    return list(missing)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "UP"}), 200
       

# Define API route
@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No input data provided"}), 400
        
        missing = feature_validation(json_data)
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        predictions = predict(json_data)
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
