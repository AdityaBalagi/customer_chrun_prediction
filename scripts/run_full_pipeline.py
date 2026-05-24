import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.prepare_data import main as prepare_data_main
from src.models.train_model import train_model
from src.evaluation.evaluate import evaluate_model


def main() -> None:
    print("=== Preparing processed data ===")
    prepare_data_main()

    print("\n=== Training the XGBoost model ===")
    train_model()

    print("\n=== Evaluating saved model ===")
    evaluate_model()


if __name__ == "__main__":
    main()
