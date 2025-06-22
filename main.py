from fastapi import FastAPI, Body, WebSocket
from datetime import date
from pydantic import BaseModel

app = FastAPI()

MODEL = None
METRICS = {
    "accuracy": None,
    "feature_importance": None,
    "last_trained": None
}

class FeatureInput(BaseModel):
    model_type: str
    symbol: str
    features: dict


@app.get("/")
def root():
    return {"message": " Predictor API is running ✅"}

@app.get("/data")
async def get_data(symbol: str, from_date: date, to_date: date):
    pass

@app.post("/predict")
def predict_endpoint(data: FeatureInput = Body(...)):
    print("📦 Input received:", data)

@app.post("/predict_all_models")
def predict_all_models_endpoint(data: FeatureInput = Body(...)):
    pass

@app.get("/create_train_data")
def create_train_data(symbol: str, from_date: date, to_date: date):
    pass

@app.get("/train_model")
def train_model_endpoint(symbol: str, from_date: date, to_date: date, model_type: str):
    pass

@app.get("/train_all_models")
def train_all_models_endpoint(symbol: str, from_date: date, to_date: date):
    pass

@app.get("/backtest_models")
def backtest_models_endpoint(symbol: str, from_date: date, to_date: date):
    pass

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    pass

@app.get("/train")
async def train(symbol: str, from_date: date, to_date: date):
    pass

@app.get("/metrics")
def get_metrics():
    pass