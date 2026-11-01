import pytest


def test_health_reports_status_and_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_version"], str)


def test_predict_returns_contract_fields_with_correct_types(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "probability", "risk_level", "model_version"}
    assert body["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(body["probability"], float)
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["model_version"], str)


def test_predict_model_version_matches_health(client, valid_payload):
    health = client.get("/health").json()
    prediction = client.post("/predict", json=valid_payload).json()
    assert prediction["model_version"] == health["model_version"]


def test_predict_without_total_charges_is_accepted(client, valid_payload):
    payload = {**valid_payload, "tenure": 0}
    payload.pop("TotalCharges")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_invalid_example_is_rejected_and_names_the_offending_fields(client, invalid_payload):
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    offending = {error["loc"][-1] for error in response.json()["detail"]}
    assert offending == {"tenure", "Contract", "MonthlyCharges"}


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", -1),
        ("tenure", 73),
        ("MonthlyCharges", 0),
        ("MonthlyCharges", "abc"),
        ("gender", "Other"),
        ("SeniorCitizen", 2),
        ("PaymentMethod", "Cash"),
    ],
)
def test_values_outside_the_contract_are_rejected(client, valid_payload, field, value):
    response = client.post("/predict", json={**valid_payload, field: value})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client, valid_payload):
    valid_payload.pop("Contract")
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_unknown_field_is_rejected(client, valid_payload):
    response = client.post("/predict", json={**valid_payload, "customerID": "1234-ABCD"})
    assert response.status_code == 422


def test_empty_body_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_internal_failure_returns_500_without_leaking_details(client, valid_payload, monkeypatch):
    def fail(features):
        raise RuntimeError("detalle interno secreto")

    monkeypatch.setattr(client.app.state.predictor, "predict", fail)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 500
    assert "detalle interno secreto" not in response.text


def test_swagger_documentation_is_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths and "/predict" in paths
