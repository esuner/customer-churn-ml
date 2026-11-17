"""Análisis de data drift con Evidently: histórico (referencia) vs lote de producción simulado."""
import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.data.load_data import load_raw_data, split_data
from src.features.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.inference.predictor import ChurnPredictor
from src.inference.risk import risk_level

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_PATH = PROJECT_ROOT / "data" / "production" / "customer_churn_current.csv"
REPORT_DIR = PROJECT_ROOT / "monitoring" / "reports"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DECISION_THRESHOLD = 0.5


def load_current(path: Path = CURRENT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES, categorical_columns=CATEGORICAL_FEATURES
    )
    ref_ds = Dataset.from_pandas(reference[FEATURES], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[FEATURES], data_definition=definition)
    return Report([DataDriftPreset()]).run(cur_ds, ref_ds)


def column_drift(snapshot) -> pd.DataFrame:
    """Una fila por variable: método, valor, umbral y si hay drift.

    Los métodos de distancia (Wasserstein, Jensen-Shannon) marcan drift si el valor supera el
    umbral; los tests estadísticos (p-value) lo marcan si el valor queda por debajo.
    """
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if not config["type"].endswith("ValueDrift"):
            continue
        score, threshold = float(metric["value"]), config["threshold"]
        is_p_value = "p_value" in config["method"]
        rows.append(
            {
                "column": config["column"],
                "method": config["method"],
                "score": round(score, 4),
                "threshold": threshold,
                "drifted": bool(score < threshold if is_p_value else score > threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def feature_weights(pipeline) -> pd.Series:
    """Peso de cada variable original en el modelo lineal: suma de |coeficiente| de sus columnas codificadas."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    weights = dict.fromkeys(FEATURES, 0.0)
    for name, coef in zip(names, coefs):
        encoded = name.split("__", 1)[1]
        for column in FEATURES:
            if encoded == column or encoded.startswith(column + "_"):
                weights[column] += abs(float(coef))
                break
    return pd.Series(weights).sort_values(ascending=False)


def prediction_summary(pipeline, frame: pd.DataFrame) -> dict:
    proba = pipeline.predict_proba(frame[FEATURES])[:, 1]
    levels = pd.Series([risk_level(p) for p in proba])
    return {
        "rows": int(len(frame)),
        "mean_probability": round(float(proba.mean()), 4),
        "share_predicted_churn": round(float((proba >= DECISION_THRESHOLD).mean()), 4),
        "share_low": round(float((levels == "LOW").mean()), 4),
        "share_medium": round(float((levels == "MEDIUM").mean()), 4),
        "share_high": round(float((levels == "HIGH").mean()), 4),
    }


def main() -> None:
    historical = load_raw_data()
    current = load_current()
    pipeline = ChurnPredictor().pipeline

    snapshot = run_drift_report(historical, current)
    drift = column_drift(snapshot)
    weights = feature_weights(pipeline)
    drift["model_weight"] = drift["column"].map(weights).round(3)

    _, X_test, _, _ = split_data(historical)
    summary = {
        "reference": {"source": "customer_churn_historical.csv", "rows": int(len(historical))},
        "current": {"source": "customer_churn_current.csv", "rows": int(len(current))},
        "drifted_columns": int(drift["drifted"].sum()),
        "total_columns": int(len(drift)),
        "columns": drift.to_dict(orient="records"),
        "predictions_reference_holdout": prediction_summary(pipeline, X_test),
        "predictions_current": prediction_summary(pipeline, current),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_DIR / "drift_report.html"))
    (REPORT_DIR / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Drift en {summary['drifted_columns']} de {summary['total_columns']} variables\n")
    print(drift.to_string(index=False))
    print("\nPredicciones del modelo (referencia = test hold-out vs producción)")
    print(pd.DataFrame(
        {"referencia": summary["predictions_reference_holdout"], "produccion": summary["predictions_current"]}
    ).to_string())
    print(f"\nReporte: {REPORT_DIR / 'drift_report.html'}")


if __name__ == "__main__":
    main()
