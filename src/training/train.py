"""Entrenamiento y comparación de modelos de churn."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.load_data import RANDOM_STATE, load_raw_data, split_data
from src.evaluation.metrics import evaluate
from src.features.preprocessing import build_preprocessor


def build_models() -> dict:
    return {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def train_and_evaluate() -> None:
    X_train, X_test, y_train, y_test = split_data(load_raw_data())
    for name, estimator in build_models().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        print(name, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    train_and_evaluate()