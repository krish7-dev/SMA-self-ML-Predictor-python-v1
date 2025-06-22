# utils/one_year_data.py
from datetime import timedelta, date
import httpx
import pandas as pd
import os
import urllib.parse

async def one_year_data(symbol: str, from_date: date, to_date: date):
    current_start = from_date
    all_candles = []
    print("Starting One Year Data","from",from_date,"to",to_date)
    async with httpx.AsyncClient() as client:
        while current_start < to_date:
            current_end = min(current_start + timedelta(days=30), to_date)
            encoded_symbol = urllib.parse.quote(symbol, safe='')
            url = f"http://localhost:7070/api/history?symbol={encoded_symbol}&from={current_start}&to={current_end}"
            print(url)
            response = await client.get(url)
            if response.status_code == 200:
                candles = response.json()
                all_candles.extend(candles)
                print(f"✅ Collected {len(candles)} candles from {current_start} to {current_end}")
            else:
                print(f"❌ Failed to get data from {current_start} to {current_end}: {response.status_code}")
            current_start = current_end + timedelta(days=1)

    file_path = None
    if all_candles:
        df = pd.DataFrame(all_candles)
        os.makedirs("data", exist_ok=True)
        # Replace | and other illegal characters in filename
        safe_symbol = symbol.replace("|", "_").replace(":", "_").replace("/", "_")
        file_path = f"data/{safe_symbol}_{from_date}_{to_date}.csv"
        df.to_csv(file_path, index=False)
        print(f"📁 Saved to {file_path}")

    return file_path, all_candles
