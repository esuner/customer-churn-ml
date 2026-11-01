"""Exporta el modelo registrado (alias 'candidate') a models/ para que la API lo cargue desde disco."""
import json
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

MODEL_NAME = "churn-classifier"
ALIAS = "candidate"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / MODEL_NAME


def main() -> None:
    load_dotenv()
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="model", dst_path=str(OUTPUT_DIR)
    )

    info = {
        "model_name": MODEL_NAME,
        "model_version": version.version,
        "source_run_id": version.run_id,
        "git_commit": version.tags.get("git_commit"),
        "data_version": version.tags.get("data_version"),
    }
    (OUTPUT_DIR / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Exportado {MODEL_NAME} v{version.version} a {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
# Cortes elegidos con el churn real observado por rango de probabilidad (ver README)
LOW_MAX = 0.30
HIGH_MIN = 0.60


def risk_level(probability: float) -> str:
    if probability < LOW_MAX:
        return "LOW"
    if probability < HIGH_MIN:
        return "MEDIUM"
    return "HIGH"
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
        probability = float(self.pipeline.predict_proba(pd.DataFrame([features]))[0, 1])
        return {
            "prediction": "CHURN" if probability >= DECISION_THRESHOLD else "NO_CHURN",
            "probability": round(probability, 4),
            "risk_level": risk_level(probability),
            "model_version": self.model_version,
        }


if __name__ == "__main__":
    example_path = PROJECT_ROOT / "data" / "examples" / "predict_request_valid.json"
    print(ChurnPredictor().predict(json.loads(example_path.read_text())))
    