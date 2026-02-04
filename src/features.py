import pandas as pd

FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek

    X = df[FEATURE_COLS]
    return X
