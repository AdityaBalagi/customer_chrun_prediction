import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS_PATH = BASE_DIR.parent / "models" / "feature_columns.json"

NUMERIC_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "num_logins_last_month",
    "avg_session_minutes",
    "payment_online",
]
CATEGORICAL_COLUMNS = ["gender", "contract_type", "payment_method"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    missing_total = df["total_charges"].isna()
    df.loc[missing_total, "total_charges"] = (
        df.loc[missing_total, "monthly_charges"] * df.loc[missing_total, "tenure_months"]
    )
    df["avg_session_minutes"] = pd.to_numeric(df["avg_session_minutes"], errors="coerce").fillna(df["avg_session_minutes"].median())
    df["support_tickets"] = pd.to_numeric(df["support_tickets"], errors="coerce").fillna(0).astype(int)
    df["tenure_months"] = pd.to_numeric(df["tenure_months"], errors="coerce").fillna(0).astype(int)
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce").fillna(df["monthly_charges"].median())
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median()).astype(int)
    if "churned" in df.columns:
        df["churned"] = df["churned"].astype(int)

    df["avg_monthly_charge"] = df["total_charges"] / df["tenure_months"].replace(0, 1)
    df["high_value"] = (df["monthly_charges"] > df["monthly_charges"].median()).astype(int)
    df["support_ticket_rate"] = df["support_tickets"] / df["tenure_months"].replace(0, 1)
    df["payment_online"] = df["payment_method"].isin(["Credit card", "Bank transfer"]).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    numeric = df[NUMERIC_COLUMNS + ["avg_monthly_charge", "high_value", "support_ticket_rate"]]
    categorical = df[CATEGORICAL_COLUMNS].copy()
    categorical = pd.get_dummies(categorical, dummy_na=False)

    features = pd.concat([numeric, categorical], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def save_feature_columns(feature_df: pd.DataFrame) -> None:
    columns = feature_df.columns.tolist()
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as fp:
        json.dump(columns, fp, indent=2)


def load_feature_columns() -> list[str]:
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    expected = load_feature_columns()
    feature_df = build_features(df)
    return feature_df.reindex(columns=expected, fill_value=0)


def prepare_processed_dataset(raw_path: Path, processed_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    raw = raw.copy()
    raw["total_charges"] = pd.to_numeric(raw["total_charges"], errors="coerce")
    feature_df = build_features(raw)
    save_feature_columns(feature_df)
    output = pd.concat([raw[["user_id", "churned"]].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(processed_path, index=False)
    return output
