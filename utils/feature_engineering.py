import pandas as pd

def generate_features(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {file_path}")

    # Sort by timestamp if needed
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Feature 1: % change
    df["pct_change"] = (df["close"] - df["open"]) / df["open"]

    # Feature 2: Moving averages (SMA)
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()

    # Feature 3: Lag features
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_2"] = df["close"].shift(2)

    # Label: 1 if next candle's close > current close
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop rows with NaN (from rolling and shifting)
    df = df.dropna().reset_index(drop=True)

    return df

def prepare_single_feature_dict(candle: dict) -> dict:
    open_price = candle["open"]
    close_price = candle["close"]
    high = candle["high"]
    low = candle["low"]
    volume = candle["volume"]

    pct_change = (close_price - open_price) / open_price

    # You may set default or dummy values for rolling/lags if unavailable
    # For real-time prediction, you'd compute SMA and lags externally and pass them in
    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,
        "volume": volume,
        "pct_change": pct_change,
        "sma_5": candle.get("sma_5", close_price),          # fallback to current close
        "sma_10": candle.get("sma_10", close_price),
        "close_lag_1": candle.get("close_lag_1", close_price),
        "close_lag_2": candle.get("close_lag_2", close_price),
    }
