import os
import glob
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "train.parquet")

def _find_csv_files():
    patterns = [
        os.path.join(RAW_DIR, "**", "*.csv"),
        os.path.join(RAW_DIR, "*.CSV"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))

def _standardize_columns(df):
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    pickup = pick("lpep_pickup_datetime", "tpep_pickup_datetime", "pickup_datetime", "pickup_time")
    dropoff = pick("lpep_dropoff_datetime", "tpep_dropoff_datetime", "dropoff_datetime", "dropoff_time")
    passenger = pick("passenger_count", "passengers")
    distance = pick("trip_distance", "distance")
    total = pick("total_amount", "fare_amount")

    rename = {}
    if pickup: rename[pickup] = "pickup_datetime"
    if dropoff: rename[dropoff] = "dropoff_datetime"
    if passenger: rename[passenger] = "passenger_count"
    if distance: rename[distance] = "trip_distance"
    if total: rename[total] = "total_amount"

    df = df.rename(columns=rename)

    required = ["pickup_datetime", "dropoff_datetime", "passenger_count", "trip_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}. Available: {list(df.columns)}")

    return df

def main():
    files = _find_csv_files()
    if not files:
        raise FileNotFoundError(
            "No CSV files found under data/raw/. "
            "Make sure you copied 2019_taxi_trips.csv into data/raw/."
        )

    # pick the largest CSV (usually trip table)
    trips_csv = max(files, key=os.path.getsize)
    print(f"Using file: {trips_csv}")

    # Read only a sample first so it runs fast
    df = pd.read_csv(trips_csv, nrows=200_000)

    df = _standardize_columns(df)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime", "trip_distance", "passenger_count"])

    df["trip_duration_minutes"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # sanity filters
    df = df[(df["trip_duration_minutes"] > 0) & (df["trip_duration_minutes"] < 180)]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 8)]

    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved dataset to {OUT_PATH} | rows={len(df)}")

if __name__ == "__main__":
    main()
