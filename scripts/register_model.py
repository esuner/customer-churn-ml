"""Registra en el Model Registry de MLflow el modelo candidato elegido."""
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

EXPERIMENT_NAME = "churn-model-comparison"
RUN_NAME = "logreg_C1_balanced"
MODEL_NAME = "churn-classifier"


def main() -> None:
    load_dotenv()
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    )
    run = runs[0]

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    v = version.version

    client.set_model_version_tag(MODEL_NAME, v, "source_run_id", run.info.run_id)
    client.set_model_version_tag(MODEL_NAME, v, "git_commit", run.data.tags["git_commit"])
    client.set_model_version_tag(MODEL_NAME, v, "data_version", run.data.tags["data_version"])
    client.update_model_version(
        MODEL_NAME, v,
        description=f"Regresion logistica balanceada (C=1.0). Origen: run {RUN_NAME}.",
    )
    client.set_registered_model_alias(MODEL_NAME, "candidate", v)

    print(f"Registrado {MODEL_NAME} version {v} (alias 'candidate')")
    print(f"Run de origen: {run.info.run_id}")


if __name__ == "__main__":
    main()