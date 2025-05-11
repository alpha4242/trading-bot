import ccxt
import pandas as pd
import time
import datetime
import os

# ====== CONFIGURATION ======
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

symbol = 'PNUTUSDT'
timeframe = '1m'
ema_short_period = 9
ema_long_period = 21
quantity = 13
leverage = 10
stoploss_lookback = 4
enable_ema50_filter = False

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
is_position_open = False
current_position_type = None  # 'buy' or 'sell'


def set_leverage(symbol, leverage):
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol:
                current_leverage = float(pos['leverage'])
                if current_leverage == leverage:
                    print(f"Leverage already set to {leverage}x — skipping.")
                    return
                break
        exchange.set_leverage(leverage, symbol)
        print(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
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


def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['atr'] = tr.rolling(window=period).mean()
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


def close_position_and_wait():
    print("🔄 Attempting to close any open positions...")
    try:
        waited = 0
        max_wait = 15
        while waited < max_wait:
            positions = exchange.fetch_positions([symbol])
            open_found = False
            for pos in positions:
                if pos['symbol'] != symbol:
                    continue
                contracts = float(pos['contracts'])
                if contracts > 0:
                    open_found = True
                    side = pos['side'].lower()
                    position_idx = int(pos['info'].get('positionIdx', 0))
                    close_side = 'sell' if side == 'long' else 'buy'

                    print(f"⚠️ Closing {side.upper()} position of size {contracts}")
                    exchange.create_order(
                        symbol,
                        'market',
                        close_side,
                        contracts,
                        None,
                        {
                            'reduceOnly': True,
                            'positionIdx': position_idx
                        }
                    )
            if not open_found:
                print("✅ Position fully closed.")
                return
            waited += 1
            time.sleep(1)

        print("❗ Warning: Position might not have closed fully.")
    except Exception as e:
        print(f"[Close Position Error]: {str(e)}")


def place_trade(signal, df):
    global is_position_open, current_position_type
    current_price = df['close'].iloc[-1]
    recent_candles = df[-stoploss_lookback:]

    sl_price = recent_candles['low'].min() if signal == 'buy' else recent_candles['high'].max()
    risk = abs(current_price - sl_price)
    tp_price = current_price + 3 * risk if signal == 'buy' else current_price - 3 * risk
    qty_80 = round(quantity * 0.8, 3)
    qty_20 = quantity - qty_80

    if is_position_open:
        print("[REVERSE] New signal while position open. Closing existing...")
        close_position_and_wait()
        is_position_open = False
        time.sleep(1)

    try:
        order_type = 'buy' if signal == 'buy' else 'sell'
        position_idx = 1 if order_type == 'buy' else 2

        # 80% with TP
        exchange.create_order(symbol, 'market', order_type, qty_80, None, {
            'positionIdx': position_idx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })
        exchange.create_order(symbol, 'limit', 'sell' if order_type == 'buy' else 'buy', qty_80, round(tp_price, 4), {
            'positionIdx': position_idx,
            'reduceOnly': True
        })

        # 20% runner without TP
        exchange.create_order(symbol, 'market', order_type, qty_20, None, {
            'positionIdx': position_idx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })

        is_position_open = True
        current_position_type = signal
        print(f"✅ Executed {signal.upper()} order at {current_price:.4f}")

    except Exception as e:
        print(f"[Order Error]: {str(e)}")


def run_bot():
    print(f"\nRunning bot at {datetime.datetime.now()}")
    try:
        set_leverage(symbol, leverage)
        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_ema(df, ema_short_period, ema_long_period)
        df = calculate_atr(df)

        signal = get_ema_signal(df)
        if not signal:
            print("🟡 No crossover detected.")
            return

        if enable_ema50_filter:
            price = df['close'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            if signal == 'buy' and price < ema_50:
                print("🔹 Skipping long: price < EMA50")
                return
            elif signal == 'sell' and price > ema_50:
                print("🔹 Skipping short: price > EMA50")
                return

        avg_price = df['close'].iloc[-14:].mean()
        atr_threshold = avg_price * 0.001
        if df['atr'].iloc[-1] < atr_threshold:
            print(f"🔸 ATR too low ({df['atr'].iloc[-1]:.6f}) < threshold ({atr_threshold:.6f})")
            return

        print(f"🔔 Trade signal: {signal.upper()}")
        place_trade(signal, df)

    except Exception as e:
        print(f"[Bot Error]: {str(e)}")

# ====== LOOP ======
while True:
    run_bot()
    time.sleep(60)
