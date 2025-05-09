import ccxt
import pandas as pd
import time
import datetime
import threading
import ta

# ====== CONFIGURATION ======
import os
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")


symbol = 'PNUTUSDT'
timeframe = '1m'
ema_short_period = 5
ema_long_period = 10
quantity = 15
leverage = 10
stoploss_lookback = 4
rsi_diff_threshold = 8
enable_ema50_filter = True  # Toggle directional filter on/off

# ====== INIT EXCHANGE ======
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'linear',
        'defaultSubType': 'linear',
        'adjustForTimeDifference': True,
    }
})

exchange.load_markets()

is_long_open = False
is_short_open = False

def set_leverage(symbol, leverage):
    market = exchange.market(symbol)
    try:
        exchange.set_leverage(leverage, market['id'])
        print(f"Leverage set to {leverage}x for {symbol}")
    except ccxt.BaseError as e:
        if "leverage not modified" in str(e).lower():
            print(f"Leverage already set to {leverage}x for {symbol}")
        else:
            print(f"[Leverage Error]: {str(e)}")

def fetch_ohlcv(symbol, timeframe, limit=200):
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_ema(df, short, long):
    df['ema_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_sma'] = df['rsi'].rolling(window=period).mean()
    return df

def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['atr'] = tr.rolling(window=period).mean()
    return df

def calculate_adx(df, period=14):
    df['adx'] = ta.trend.ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period,
        fillna=False
    ).adx()
    return df

def get_ema_signal(df):
    ema_short_prev2 = df['ema_short'].iloc[-3]
    ema_long_prev2 = df['ema_long'].iloc[-3]
    ema_short_prev1 = df['ema_short'].iloc[-2]
    ema_long_prev1 = df['ema_long'].iloc[-2]

    if ema_short_prev2 < ema_long_prev2 and ema_short_prev1 > ema_long_prev1:
        return 'buy'
    elif ema_short_prev2 > ema_long_prev2 and ema_short_prev1 < ema_long_prev1:
        return 'sell'
    return None

def monitor_position(position_type):
    global is_long_open, is_short_open
    positionIdx = 1 if position_type == 'buy' else 2

    min_hold_time = 60
    check_interval = 15
    trailing_buffer = 0.0015

    entry_time = time.time()
    print(f"[MONITOR] Started monitoring {position_type.upper()} position...")
    time.sleep(min_hold_time)

    entry_df = fetch_ohlcv(symbol, timeframe)
    entry_price = entry_df['close'].iloc[-1]

    while True:
        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_rsi(df)

        rsi = df['rsi'].iloc[-1]
        rsi_sma = df['rsi_sma'].iloc[-1]
        diff = abs(rsi - rsi_sma)
        current_price = df['close'].iloc[-1]
        elapsed = time.time() - entry_time

        print(f"[MONITOR] Type: {position_type.upper()}, Time Held: {elapsed:.1f}s, RSI: {rsi:.2f}, SMA: {rsi_sma:.2f}, Diff: {diff:.2f}, Price: {current_price:.5f}")

        if position_type == 'buy' and rsi < rsi_sma and diff >= rsi_diff_threshold:
            print("[EXIT] RSI exit condition met for BUY")
            close_position(1)
            is_long_open = False
            break
        elif position_type == 'sell' and rsi > rsi_sma and diff >= rsi_diff_threshold:
            print("[EXIT] RSI exit condition met for SELL")
            close_position(2)
            is_short_open = False
            break

        if position_type == 'buy' and current_price < entry_price * (1 - trailing_buffer):
            print("[EXIT] Trailing stop hit for BUY")
            close_position(1)
            is_long_open = False
            break
        elif position_type == 'sell' and current_price > entry_price * (1 + trailing_buffer):
            print("[EXIT] Trailing stop hit for SELL")
            close_position(2)
            is_short_open = False
            break

        time.sleep(check_interval)

def close_position(positionIdx):
    side = 'sell' if positionIdx == 1 else 'buy'
    exchange.create_order(symbol, 'market', side, quantity, None, {'positionIdx': positionIdx})
    print(f"Closed position with {side.upper()} order.")

def place_order(signal, df):
    global is_long_open, is_short_open
    recent_candles = df[-stoploss_lookback:]
    current_price = df['close'].iloc[-1]

    if signal == 'buy' and not is_long_open:
        sl_price = recent_candles['low'].min()
        tp_price = current_price + (current_price - sl_price) * 3
        params = {
            'positionIdx': 1,
            'stopLoss': round(sl_price, 4),
            'takeProfit': round(tp_price, 4),
            'slTriggerBy': 'LastPrice',
            'tpTriggerBy': 'LastPrice'
        }
        order = exchange.create_market_buy_order(symbol, quantity, params)
        is_long_open = True
        print(f"Executed BUY order: {order['id']}")
        threading.Thread(target=monitor_position, args=('buy',)).start()

    elif signal == 'sell' and not is_short_open:
        sl_price = recent_candles['high'].max()
        tp_price = current_price - (sl_price - current_price) * 3
        params = {
            'positionIdx': 2,
            'stopLoss': round(sl_price, 4),
            'takeProfit': round(tp_price, 4),
            'slTriggerBy': 'LastPrice',
            'tpTriggerBy': 'LastPrice'
        }
        order = exchange.create_market_sell_order(symbol, quantity, params)
        is_short_open = True
        print(f"Executed SELL order: {order['id']}")
        threading.Thread(target=monitor_position, args=('sell',)).start()

def run_bot():
    print(f"\nRunning bot at {datetime.datetime.now()}")
    try:
        set_leverage(symbol, leverage)
        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_ema(df, ema_short_period, ema_long_period)
        df = calculate_rsi(df)
        df = calculate_atr(df)
        df = calculate_adx(df)
        print(f"Latest ADX: {df['adx'].iloc[-1]:.2f}")

        average_price = df['close'].iloc[-14:].mean()
        dynamic_atr_threshold = average_price * 0.001

        signal = get_ema_signal(df)
        if not signal:
            print("ANALYSING THE MARKET")
            return

        # Directional logic based on EMA 50
        if enable_ema50_filter:
            price = df['close'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]

            if signal == 'buy' and price < ema_50:
                print("Skipping long because market is bearish (price < EMA50)")
                return
            elif signal == 'sell' and price > ema_50:
                print("Skipping short because market is bullish (price > EMA50)")
                return

        if df['adx'].iloc[-1] < 20:
            print(f"Weak trend (ADX: {df['adx'].iloc[-1]:.2f}) - skipping.")
            return

        if df['atr'].iloc[-1] < dynamic_atr_threshold:
            print(f"Low ATR ({df['atr'].iloc[-1]:.6f}) below threshold ({dynamic_atr_threshold:.6f}) - skipping.")
            return

        print(f"Confirmed signal: {signal.upper()}")
        place_order(signal, df)

    except Exception as e:
        print(f"[Bot Error]: {str(e)}")

# ====== LOOP ======
while True:
    run_bot()
    time.sleep(60)