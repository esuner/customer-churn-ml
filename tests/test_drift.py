import numpy as np
import pandas as pd
import pytest

from monitoring.drift_report import (
    FEATURES,
    column_drift,
    feature_weights,
    prediction_summary,
    run_drift_report,
)
from src.features.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.inference.predictor import ChurnPredictor


def synthetic_frame(rows: int, seed: int, tenure_shift: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "tenure": rng.integers(0, 40, rows) + tenure_shift,
        "MonthlyCharges": rng.uniform(20, 110, rows),
        "TotalCharges": rng.uniform(0, 6000, rows),
    }
    for column in CATEGORICAL_FEATURES:
        data[column] = rng.choice(["Yes", "No"], rows)
    return pd.DataFrame(data)[FEATURES]


@pytest.fixture(scope="module")
def drift_table() -> pd.DataFrame:
    reference = synthetic_frame(600, seed=1)
    current = synthetic_frame(600, seed=2, tenure_shift=30)
    return column_drift(run_drift_report(reference, current))


def test_drift_table_has_one_row_per_feature(drift_table):
    assert set(drift_table["column"]) == set(FEATURES)
    assert {"method", "score", "threshold", "drifted"} <= set(drift_table.columns)


def test_injected_shift_is_flagged_and_stable_columns_are_not(drift_table):
    drifted = set(drift_table.loc[drift_table["drifted"], "column"])
    assert "tenure" in drifted
    assert "gender" not in drifted


def test_feature_weights_cover_every_feature_and_are_non_negative():
    weights = feature_weights(ChurnPredictor().pipeline)
    assert set(weights.index) == set(FEATURES)
    assert (weights >= 0).all()
    assert weights.loc[NUMERIC_FEATURES].sum() > 0


def test_prediction_summary_shares_are_consistent():
    summary = prediction_summary(ChurnPredictor().pipeline, synthetic_frame(200, seed=3))
    assert summary["rows"] == 200
    assert 0 <= summary["mean_probability"] <= 1
    assert summary["share_low"] + summary["share_medium"] + summary["share_high"] == pytest.approx(1.0, abs=1e-3)
