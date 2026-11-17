import json
import logging
import uuid

from app.logging_config import JsonFormatter


def test_every_response_carries_a_generated_request_id(client):
    response = client.get("/health")
    request_id = response.headers["X-Request-ID"]
    assert uuid.UUID(request_id)
    assert float(response.headers["X-Process-Time-ms"]) >= 0


def test_each_request_gets_a_different_request_id(client):
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]
    assert first != second


def test_client_supplied_request_id_is_propagated_to_header_and_body(client, valid_payload):
    response = client.post("/predict", json=valid_payload, headers={"X-Request-ID": "demo-123"})
    assert response.headers["X-Request-ID"] == "demo-123"
    assert response.json()["request_id"] == "demo-123"


def test_unsafe_request_id_is_replaced(client):
    response = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert uuid.UUID(response.headers["X-Request-ID"])


def test_request_log_has_status_duration_and_model_version(client, valid_payload, caplog):
    with caplog.at_level(logging.INFO, logger="churn-api"):
        client.post("/predict", json=valid_payload, headers={"X-Request-ID": "log-check"})
    request_logs = [r for r in caplog.records if r.getMessage() == "request"]
    assert request_logs
    fields = request_logs[-1].fields
    assert fields["path"] == "/predict"
    assert fields["status_code"] == 200
    assert fields["duration_ms"] >= 0
    assert fields["model_version"]


def test_prediction_log_does_not_contain_customer_data(client, valid_payload, caplog):
    with caplog.at_level(logging.INFO, logger="churn-api"):
        client.post("/predict", json=valid_payload)
    prediction_logs = [r for r in caplog.records if r.getMessage() == "prediction"]
    assert prediction_logs
    assert set(prediction_logs[-1].fields) == {"prediction", "probability", "risk_level", "model_version"}


def test_errors_are_logged_with_their_status_code(client, invalid_payload, caplog):
    with caplog.at_level(logging.INFO, logger="churn-api"):
        client.post("/predict", json=invalid_payload)
    request_logs = [r for r in caplog.records if r.getMessage() == "request"]
    assert request_logs[-1].fields["status_code"] == 422


def test_json_formatter_emits_one_parseable_json_line():
    record = logging.LogRecord("churn-api", logging.INFO, __file__, 1, "request", None, None)
    record.fields = {"status_code": 200}
    parsed = json.loads(JsonFormatter().format(record))
    assert parsed["message"] == "request"
    assert parsed["status_code"] == 200
    assert {"timestamp", "level", "logger", "request_id"} <= set(parsed)
