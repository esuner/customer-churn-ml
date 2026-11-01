import pytest

from src.inference.predictor import ChurnPredictor


@pytest.fixture(scope="module")
def predictor() -> ChurnPredictor:
    return ChurnPredictor()


@pytest.fixture
def loyal_payload(valid_payload) -> dict:
    return {
        **valid_payload,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 25.0,
        "TotalCharges": 1800.0,
    }


def test_model_loads_with_traceability_metadata(predictor):
    assert hasattr(predictor.pipeline, "predict_proba")
    assert predictor.model_version.isdigit()
    for key in ("source_run_id", "git_commit", "data_version"):
        assert predictor.info[key]


def test_prediction_has_expected_structure_and_types(predictor, valid_payload):
    result = predictor.predict(valid_payload)
    assert set(result) == {"prediction", "probability", "risk_level", "model_version"}
    assert result["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["model_version"] == predictor.model_version


def test_prediction_is_deterministic(predictor, valid_payload):
    assert predictor.predict(valid_payload) == predictor.predict(valid_payload)


def test_churn_prone_customer_scores_higher_than_loyal_customer(predictor, valid_payload, loyal_payload):
    risky = predictor.predict(valid_payload)
    loyal = predictor.predict(loyal_payload)
    assert risky["probability"] > loyal["probability"]
    assert risky["risk_level"] == "HIGH"
    assert loyal["risk_level"] == "LOW"


def test_missing_total_charges_is_imputed(predictor, valid_payload):
    result = predictor.predict({**valid_payload, "tenure": 0, "TotalCharges": None})
    assert 0.0 <= result["probability"] <= 1.0
