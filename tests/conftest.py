import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "examples"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return json.loads((EXAMPLES / "predict_request_valid.json").read_text())


@pytest.fixture
def invalid_payload() -> dict:
    return json.loads((EXAMPLES / "predict_request_invalid.json").read_text())
