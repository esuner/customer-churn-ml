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
