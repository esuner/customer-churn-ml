"""Carga y partición reproducible del dataset histórico de churn."""
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "customer_churn_historical.csv"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


# TODO: split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> train_test_split
# estratificado por Churn (X_train, X_test, y_train, y_test). Ver README, sección
# "Próximos pasos" para el detalle de por qué stratify y random_state importan acá.
