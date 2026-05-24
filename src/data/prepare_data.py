from pathlib import Path

from src.features.feature_engineering import prepare_processed_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "sample_churn_data.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_churn_data.csv"


def main() -> None:
    print(f"Loading raw data from {RAW_DATA_PATH}")
    processed = prepare_processed_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")
    print(f"Processed rows: {len(processed)}")


if __name__ == "__main__":
    main()
