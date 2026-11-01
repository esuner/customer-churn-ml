"""Carga del modelo exportado y predicción de churn."""
import json
import pickle
from pathlib import Path

import pandas as pd

from src.inference.risk import risk_level

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "churn-classifier"
DECISION_THRESHOLD = 0.5


class ChurnPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        with open(model_dir / "model" / "model.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        self.info = json.loads((model_dir / "model_info.json").read_text())

    @property
    def model_version(self) -> str:
        return self.info["model_version"]

    def predict(self, features: dict) -> dict:
        frame = pd.DataFrame([features])
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        return {
            "prediction": "CHURN" if probability >= DECISION_THRESHOLD else "NO_CHURN",
            "probability": round(probability, 4),
            "risk_level": risk_level(probability),
            "model_version": self.model_version,
        }


if __name__ == "__main__":
    example_path = PROJECT_ROOT / "data" / "examples" / "predict_request_valid.json"
    print(ChurnPredictor().predict(json.loads(example_path.read_text())))
