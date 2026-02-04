import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features import build_features

DATA_PATH = "data/train.parquet"
MODEL_PATH = "models/model.joblib"

def train():
    df = pd.read_parquet(DATA_PATH)

    X = build_features(df)
    y = df["trip_duration_minutes"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 100,
        "max_depth": 12,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)

    mlflow.set_experiment("trip-duration-mlops")

    with mlflow.start_run():
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_val, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, artifact_path="model")

        joblib.dump(model, MODEL_PATH)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print("Model saved to models/model.joblib")

if __name__ == "__main__":
    train()
