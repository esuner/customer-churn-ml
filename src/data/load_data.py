"""Carga y partición reproducible del dataset histórico de churn."""
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "customer_churn_historical.csv"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Churn rate train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")
