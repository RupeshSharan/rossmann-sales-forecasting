import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    data_path = os.path.join(BASE_DIR, "..", "data", "processed", "rossmann_features.csv")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def load_sarima():
    model_path = os.path.join(BASE_DIR, "..", "models", "sarima", "sarima_model.pkl")
    return joblib.load(model_path)

def load_prophet():
    model_path = os.path.join(BASE_DIR, "..", "models", "prophet", "prophet_model.pkl")
    return joblib.load(model_path)
