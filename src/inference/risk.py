"""Conversión de la probabilidad de churn en un nivel de riesgo."""

# Cortes elegidos con el churn real observado por rango de probabilidad (ver README)
LOW_MAX = 0.30
HIGH_MIN = 0.60


def risk_level(probability: float) -> str:
    if probability < LOW_MAX:
        return "LOW"
    if probability < HIGH_MIN:
        return "MEDIUM"
    return "HIGH"
