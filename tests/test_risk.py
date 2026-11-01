import pytest

from src.inference.risk import risk_level


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, "LOW"),
        (0.299, "LOW"),
        (0.30, "MEDIUM"),
        (0.599, "MEDIUM"),
        (0.60, "HIGH"),
        (1.0, "HIGH"),
    ],
)
def test_risk_level_boundaries(probability, expected):
    assert risk_level(probability) == expected
