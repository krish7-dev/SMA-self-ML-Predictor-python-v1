import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Define consistent feature set
FEATURES = [
    "open", "high", "low", "close", "volume",
    "pct_change", "sma_5", "sma_10", "close_lag_1", "close_lag_2"
]

def save_feature_schema(symbol: str, model_type: str):
    schema_path = f"models/{symbol.replace('|', '_')}_{model_type}_features.json"
    with open(schema_path, "w") as f:
        json.dump(FEATURES, f)

def train_model(symbol: str, from_date: str, to_date: str, model_type: str):
    file_name = f"training_data/{symbol.replace('|', '_')}_{from_date}_{to_date}_train.csv"
    df = pd.read_csv(file_name)

    X = df[FEATURES]
    y = df["label"]

    if model_type == "logistic":
        model = LogisticRegression(max_iter=3000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "xgboost":
        model = XGBClassifier(eval_metric='logloss')
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X, y)
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{symbol.replace('|', '_')}_{model_type}.pkl"
    joblib.dump(model, model_path)
    save_feature_schema(symbol, model_type)

    return {"accuracy": accuracy, "model_path": model_path}

def train_all_models(symbol: str, from_date: str, to_date: str):
    file_name = f"training_data/{symbol.replace('|', '_')}_{from_date}_{to_date}_train.csv"
    df = pd.read_csv(file_name)

    X = df[FEATURES]
    y = df["label"]

    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "xgboost": XGBClassifier(eval_metric='logloss'),
        "lightgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier(verbose=0)
    }

    os.makedirs("models", exist_ok=True)
    results = {}

    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = accuracy_score(y, preds)
        model_path = f"models/{symbol.replace('|', '_')}_{name}.pkl"
        joblib.dump(model, model_path)
        save_feature_schema(symbol, name)
        results[name] = {
            "accuracy": accuracy,
            "model_path": model_path
        }

    return results

def backtest_models(symbol: str, from_date: str, to_date: str):
    file_path = f"training_data/{symbol.replace('|', '_')}_{from_date}_{to_date}_train.csv"
    df = pd.read_csv(file_path)

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "logistic": LogisticRegression(max_iter=3000),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "xgboost": XGBClassifier(eval_metric='logloss'),
        "lightgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier(verbose=0)
    }

    report = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report[name] = classification_report(y_test, preds, output_dict=True)

    return report

def predict_with_model(model_type: str, symbol: str, features: dict, return_proba=False):
    model_path = f"models/{symbol.replace('|', '_')}_{model_type}.pkl"
    features_path = f"models/{symbol.replace('|', '_')}_{model_type}_features.json"

    if not os.path.exists(model_path):
        raise ValueError("Model not found. Train it first.")

    if not os.path.exists(features_path):
        raise ValueError("Feature schema not found. Train the model again.")

    with open(features_path, "r") as f:
        expected_features = json.load(f)

    # Order features and fill missing with 0
    row = [features.get(col, 0) for col in expected_features]
    X = pd.DataFrame([row], columns=expected_features)

    model = joblib.load(model_path)
    prediction = model.predict(X)[0]

    if return_proba and hasattr(model, "predict_proba"):
        confidence = model.predict_proba(X)[0][prediction]
        return int(prediction), float(confidence)

    return int(prediction), None
