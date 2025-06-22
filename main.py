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

#Fetches and stores historical candle data for a symbol between two dates.
@app.get("/data")
async def get_data(symbol: str, from_date: date, to_date: date):
    pass

#Makes prediction using (REST) a specified model and feature set.
@app.post("/predict")
def predict_endpoint(data: FeatureInput = Body(...)):
    print("📦 Input received:", data)

#Makes prediction using (Web Socket) a specified model and feature set.
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    pass

#Predicts using all five model types and writes results to a local JSON + opens HTML.
@app.post("/predict_all_models")
def predict_all_models_endpoint(data: FeatureInput = Body(...)):
    pass

#Converts raw candle data to engineered features and saves as training dataset.
@app.get("/create_train_data")
def create_train_data(symbol: str, from_date: date, to_date: date):
    pass

#Trains a single model (model_type) for given symbol + date range.
@app.get("/train_model")
def train_model_endpoint(symbol: str, from_date: date, to_date: date, model_type: str):
    pass

#Trains all supported models for a given symbol and date range.
@app.get("/train_all_models")
def train_all_models_endpoint(symbol: str, from_date: date, to_date: date):
    pass

#Backtests all trained models on historical data.
@app.get("/backtest_models")
def backtest_models_endpoint(symbol: str, from_date: date, to_date: date):
    pass



# @app.get("/train")
# async def train(symbol: str, from_date: date, to_date: date):
#     pass

# @app.get("/metrics")
# def get_metrics():
#     pass