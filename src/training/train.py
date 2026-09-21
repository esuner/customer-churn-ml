"""Entrenamiento y comparación de modelos de churn con tracking en MLflow."""
import re
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.load_data import RANDOM_STATE, load_raw_data, split_data
from src.evaluation.metrics import evaluate
from src.features.preprocessing import build_preprocessor

EXPERIMENT_NAME = "churn-model-comparison"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DVC_FILE = PROJECT_ROOT / "data" / "raw" / "customer_churn_historical.csv.dvc"


def build_experiments() -> list:
    return [
        ("baseline_dummy", DummyClassifier(strategy="prior")),
        ("logreg_C1", LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
        ("logreg_C1_balanced", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ("logreg_C0.1_balanced", LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ("rf_200", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE)),
        ("rf_200_depth8_balanced", RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)),
        ("rf_300_depth5_balanced", RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)),
    ]


def get_git_info() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip() != ""
    return {"git_commit": commit, "git_dirty": str(dirty)}


def get_data_version() -> str:
    match = re.search(r"md5:\s*(\w+)", DVC_FILE.read_text())
    return match.group(1) if match else "unknown"


def run_experiment(name, estimator, X_train, X_test, y_train, y_test, tags) -> dict:
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=["f1", "recall", "roc_auc"])

    with mlflow.start_run(run_name=name):
        mlflow.set_tags(tags)
        mlflow.log_params({"model_type": type(estimator).__name__, **estimator.get_params()})

        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({
            "cv_f1": cv_scores["test_f1"].mean(),
            "cv_recall": cv_scores["test_recall"].mean(),
            "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
        })

        signature = infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)
    return metrics


def main() -> None:
    load_dotenv()
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = split_data(load_raw_data())
    tags = {**get_git_info(), "data_version": get_data_version()}

    for name, estimator in build_experiments():
        metrics = run_experiment(name, estimator, X_train, X_test, y_train, y_test, tags)
        print(f"{name}: f1={metrics['f1']:.3f} recall={metrics['recall']:.3f} "
              f"roc_auc={metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()