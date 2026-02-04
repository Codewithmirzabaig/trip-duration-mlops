# Trip Duration Prediction (ML + MLOps)

Predicts taxi trip duration (minutes) using a trained ML model.

## What’s inside
- Data prep: raw CSV → `data/train.parquet`
- Training: RandomForest + MLflow tracking
- API: FastAPI `/predict` endpoint
- Model saved: `models/model.joblib`

## Run locally
### 1) Create dataset
python src/make_dataset.py

### 2) Train model
python -m src.train

### 3) MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
Open: http://127.0.0.1:5000

### 4) Start API
uvicorn app.main:app --reload
Open docs: http://127.0.0.1:8000/docs
