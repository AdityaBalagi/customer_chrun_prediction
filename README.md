# Churn Prediction Project

## Overview
This repository contains a full customer churn prediction application for application users.
It includes sample raw data, a feature engineering pipeline, model training, saved artifacts, evaluation, and a Flask-based inference API with CSV upload/download support.

## Project Architecture
- `data/`
  - `data/raw/` — raw input CSV files for training and inference
  - `data/processed/` — cleaned and feature-engineered datasets
- `notebooks/` — optional EDA and experiment notebooks
- `src/`
  - `src/data/` — data preparation scripts
  - `src/features/` — feature engineering logic
  - `src/models/` — model training and persistence
  - `src/evaluation/` — evaluation and reporting scripts
  - `src/api/` — Flask inference API and frontend assets
  - `src/config/` — configuration helpers
  - `src/utils/` — utility helpers
- `models/` — serialized model artifacts (`.pkl` files)
- `reports/` — evaluation reports and summaries
- `scripts/` — pipeline orchestration scripts
- `tests/` — automated tests

## Folder Structure
- `data/raw/`
- `data/processed/`
- `notebooks/`
- `src/config/`
- `src/data/`
- `src/features/`
- `src/models/`
- `src/evaluation/`
- `src/api/`
- `src/utils/`
- `models/`
- `reports/`
- `scripts/`
- `tests/`

## Expected Input CSV Format
The Flask app and pipeline require a CSV file with the following columns:

- `user_id`
- `gender`
- `age`
- `tenure_months`
- `monthly_charges`
- `total_charges`
- `contract_type`
- `payment_method`
- `support_tickets`
- `num_logins_last_month`
- `avg_session_minutes`

Example header:
```csv
user_id,gender,age,tenure_months,monthly_charges,total_charges,contract_type,payment_method,support_tickets,num_logins_last_month,avg_session_minutes
```

The inference endpoint does not require `churned`; it predicts churn based on the input fields.

## Technical Stack
- Python 3.11+ (recommended)
- `pandas`, `numpy`
- `scikit-learn`, `xgboost`
- `flask`
- `joblib`
- `pytest`
- `jupyterlab` for notebooks

## Dependencies
Install all required packages from `requirements.txt`.

## Environment Setup (Windows PowerShell)
1. Open PowerShell in the project root.
2. Run the setup script to create and activate the virtual environment and install dependencies:
   ```powershell
   .\setup.ps1
   ```
3. If you need to manually upgrade pip later:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Full Pipeline Workflow
1. Place raw training data in `data/raw/`.
2. Run data preparation and feature engineering:
   ```powershell
   python .\scripts\run_full_pipeline.py
   ```
3. The pipeline will produce:
   - `data/processed/processed_churn_data.csv`
   - `models/churn_xgb_model.pkl`
   - `reports/evaluation_report.txt`
   - `data/processed/evaluation_predictions.csv`

## Running the Flask Inference API
1. Start the API server:
   ```powershell
   python -m src.api.app
   ```
2. Visit:
   ```text
   http://127.0.0.1:5000/
   ```
3. Upload a valid CSV file and download the output CSV.

## Output CSV Format
The output CSV includes:
- original input columns
- engineered feature columns
- `predicted_churn`
- `predicted_probability`

Example output columns:
`user_id,gender,age,tenure_months,monthly_charges,total_charges,contract_type,payment_method,support_tickets,num_logins_last_month,avg_session_minutes,age,tenure_months,monthly_charges,total_charges,support_tickets,num_logins_last_month,avg_session_minutes,payment_online,avg_monthly_charge,high_value,support_ticket_rate,gender_Female,gender_Male,contract_type_Month-to-month,contract_type_One year,contract_type_Two year,payment_method_Bank transfer,payment_method_Credit card,payment_method_Electronic check,payment_method_Mail check,predicted_churn,predicted_probability`

## Notes
- Use `data/raw/sample_churn_data.csv` as model training data.
- Use `data/raw/sample_churn_data_test.csv` for API inference tests.
- If you add new categorical or numeric fields, update `src/features/feature_engineering.py` and rerun the pipeline.

## Next Improvements
- Add unit tests for feature engineering and API behavior.
- Add a notebook in `notebooks/` for EDA and model diagnostics.
- Add Docker support for deployment.
