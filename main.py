import os
import time
import datetime
import hmac
import hashlib
import threading
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ====== CONFIGURATION ======
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

BASE_URL = "https://api.bybit.com"
SYMBOL = "SOLUSDT"
INTERVAL = "1"  # 15-minute candles
QUANTITY = 0.1
LEVERAGE = 100

# === BYBIT API HELPER ===
def send_signed_request(http_method, endpoint, params=None):
    if params is None:
        params = {}
    timestamp = str(int(time.time() * 1000))
    params['apiKey'] = API_KEY
    params['timestamp'] = timestamp
    params['recvWindow'] = 5000

    sorted_params = dict(sorted(params.items()))
    query_string = "&".join([f"{key}={value}" for key, value in sorted_params.items()])
    signature = hmac.new(bytes(API_SECRET, "utf-8"), bytes(query_string, "utf-8"), hashlib.sha256).hexdigest()
    sorted_params["sign"] = signature

    url = BASE_URL + endpoint
    try:
        if http_method == "GET":
            response = requests.get(url, params=sorted_params)
        else:
            response = requests.post(url, data=sorted_params)
        return response.json()
    except Exception as e:
        print("API Error:", e)
        return {}

# === FETCH OHLCV DATA ===
def get_klines(symbol, interval="15", limit=200):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        res = requests.get(url, params=params).json()
        df = pd.DataFrame(res['result']['list'])
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float,
                        'volume': float, 'turnover': float})
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print("Kline fetch error:", e)
        return pd.DataFrame()

# === HEIKIN ASHI CALCULATION ===
def heikin_ashi(df):
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'][0] + df['close'][0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + df['ha_close'][i-1]) / 2)
    df['ha_open'] = ha_open
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
    return df

# === STRATEGY LOGIC ===
def check_signal(df):
    df = heikin_ashi(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if last['ha_close'] > last['ha_open'] and prev['ha_close'] < prev['ha_open']:
        return "long", last['ha_low'], last['close']
    elif last['ha_close'] < last['ha_open'] and prev['ha_close'] > prev['ha_open']:
        return "short", last['ha_high'], last['close']
    return None, None, None

# === ORDER EXECUTION ===
def place_order(side, qty, stop_loss, take_profit):
    order_side = "Buy" if side == "long" else "Sell"
    tp = round(take_profit, 2)
    sl = round(stop_loss, 2)

    print(f"Placing {side.upper()} order. TP: {tp}, SL: {sl}")

    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "side": order_side,
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "GTC",
        "takeProfit": tp,
        "stopLoss": sl,
        "reduceOnly": False
    }
    return send_signed_request("POST", "/v5/order/create", params)

# === MAIN LOOP ===
def run_bot():
    in_position = False
    while True:
        try:
            df = get_klines(SYMBOL, INTERVAL)
            if df.empty:
                print("No data retrieved.")
                time.sleep(60)
                continue

            signal, sl, entry = check_signal(df)

            if signal and not in_position:
                risk = abs(entry - sl)
                tp = entry + (5 * risk) if signal == "long" else entry - (5 * risk)
                response = place_order(signal, QUANTITY, sl, tp)
                print("Order Response:", response)
                in_position = True
            else:
                print("No signal or already in trade.")

            # Auto-reset for demo; in production track actual position via API
            in_position = False

            time.sleep(60 * 15)  # Wait for next candle
        except Exception as e:
            print("Runtime Error:", e)
            time.sleep(60)

# === START BOT ===
if __name__ == "__main__":
    run_bot()

