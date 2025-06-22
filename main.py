from fastapi import FastAPI, Body, WebSocket
from datetime import date
from pydantic import BaseModel
import json, webbrowser

from utils.feature_engineering import *
from utils.model_trainer import *
from utils.one_year_data import *

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
    print(symbol,from_date,to_date)
    file_path, candles = await one_year_data(symbol, from_date, to_date)
    print(file_path,candles[:5])
    if file_path and candles:
        return {
            "message": f"Data collected and saved to {file_path}",
            "preview": candles[:5]
        }
    else:
        return {"error": "No data collected or saving failed."}

#Makes prediction using (REST) a specified model and feature set.
@app.post("/predict")
def predict_endpoint(data: FeatureInput = Body(...)):
    print("📦 Input received:", data)
    try:
        result, confidence = predict_with_model(data.model_type, data.symbol, data.features, return_proba=True)
        return {
            "prediction": result,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"error": str(e)}

#Makes prediction using (Web Socket) a specified model and feature set.
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected for predictions")

    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)

            model_type = data["model_type"]
            symbol = data["symbol"]
            candle = data["candle"]

            features = prepare_single_feature_dict(candle)  # You can use your own logic here

            result, confidence = predict_with_model(model_type, symbol, features, return_proba=True)

            response = {
                "prediction": result,
                "confidence": round(confidence, 4),
                "timestamp": candle.get("timestamp")
            }

            await websocket.send_text(json.dumps(response))

        except Exception as e:
            await websocket.send_text(json.dumps({"error": str(e)}))

#Predicts using all five model types and writes results to a local JSON + opens HTML.
@app.post("/predict_all_models")
def predict_all_models_endpoint(data: FeatureInput = Body(...)):
    models = ["logistic", "random_forest", "xgboost", "lightgbm", "catboost"]
    results = {}
    for model_type in models:
        try:
            prediction, confidence = predict_with_model(model_type, data.symbol, data.features, return_proba=True)
            results[model_type] = {
                "prediction": prediction,
                "confidence": round(confidence, 4) if confidence is not None else None
            }
        except Exception as e:
            results[model_type] = {"error": str(e)}
        # Save results to file
    with open("prediction_output.json", "w") as f:
        json.dump(results, f, indent=4)

    # Open viewer
    webbrowser.open("file://" + os.path.abspath("prediction_view.html"))
    return results

#Converts raw candle data to engineered features and saves as training dataset.
@app.get("/create_train_data")
def create_train_data(symbol: str, from_date: date, to_date: date):
    input_path = f"data/{symbol.replace('|', '_')}_{from_date}_{to_date}.csv"
    output_path = f"training_data/{symbol.replace('|', '_')}_{from_date}_{to_date}_train.csv"

    try:
        df = generate_features(input_path)
        df.to_csv(output_path, index=False)
        return {
            "message": f"Training data saved to {output_path}",
            "preview": df.head(5).to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}

#Trains a single model (model_type) for given symbol + date range.
@app.get("/train_model")
def train_model_endpoint(symbol: str, from_date: date, to_date: date, model_type: str):
    try:
        result = train_model(symbol, str(from_date), str(to_date), model_type)
        return {"message": f"{model_type} model trained", "metrics": result}
    except Exception as e:
        return {"error": str(e)}

#Trains all supported models for a given symbol and date range.
@app.get("/train_all_models")
def train_all_models_endpoint(symbol: str, from_date: date, to_date: date):
    try:
        result = train_all_models(symbol, str(from_date), str(to_date))
        return {
            "message": "All models trained successfully.",
            "metrics": result
        }
    except Exception as e:
        return {"error": str(e)}

#Backtests all trained models on historical data.
@app.get("/backtest_models")
def backtest_models_endpoint(symbol: str, from_date: date, to_date: date):
    try:
        result = backtest_models(symbol, str(from_date), str(to_date))
        return {"message": "Backtest complete", "results": result}
    except Exception as e:
        return {"error": str(e)}



# @app.get("/train")
# async def train(symbol: str, from_date: date, to_date: date):
#     pass

# @app.get("/metrics")
# def get_metrics():
#     pass