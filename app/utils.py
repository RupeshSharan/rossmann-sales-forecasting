import pandas as pd
import joblib
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent

def load_data():
    return pd.read_csv(
        BASE_PATH / "data/processed/rossmann_features.csv",
        parse_dates=["Date"]
    )

def load_sarima():
    return joblib.load(
        BASE_PATH / "models/sarima/sarima_model.pkl"
    )

def load_prophet():
    return joblib.load(
        BASE_PATH / "models/prophet/prophet_model.pkl"
    )
