import ccxt
import pandas as pd
import time
import datetime
import os

# ====== CONFIGURATION ======
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

symbol = 'PNUTUSDT'  # Use correct CCXT format
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
market = exchange.market(symbol)

# ====== STATE TRACKING ======
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


def close_position():
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] != symbol:
                continue
            size = float(pos['contracts'])
            if size > 0:
                side = pos['side'].lower()
                close_side = 'sell' if side == 'long' else 'buy'
                position_idx = int(pos['info'].get('positionIdx', 0))
                exchange.create_order(symbol, 'market', close_side, size, None, {
                    'reduceOnly': True,
                    'positionIdx': position_idx
                })
                print(f"Closed {side.upper()} position of size {size}")
                time.sleep(1)
    except Exception as e:
        print(f"[Close Error]: {str(e)}")


def place_trade(signal, df):
    global current_position_type

    current_price = df['close'].iloc[-1]
    recent_candles = df[-stoploss_lookback:]

    sl_price = recent_candles['low'].min() if signal == 'buy' else recent_candles['high'].max()
    risk = abs(current_price - sl_price)
    tp_price = current_price + 3 * risk if signal == 'buy' else current_price - 3 * risk

    qty_80 = exchange.amount_to_precision(symbol, quantity * 0.8)
    qty_20 = exchange.amount_to_precision(symbol, quantity * 0.2)

    order_side = 'buy' if signal == 'buy' else 'sell'
    position_idx = 1 if order_side == 'buy' else 2

    try:
        print(f"Placing {signal.upper()} trade - Entry: {current_price:.4f}, SL: {sl_price:.4f}, TP: {tp_price:.4f}")

        # 80% position: SL + TP
        exchange.create_order(symbol, 'market', order_side, qty_80, None, {
            'positionIdx': position_idx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })
        exchange.create_order(symbol, 'limit', 'sell' if signal == 'buy' else 'buy', qty_80, round(tp_price, 4), {
            'positionIdx': position_idx,
            'reduceOnly': True
        })

        # 20% position: SL only (runner)
        exchange.create_order(symbol, 'market', order_side, qty_20, None, {
            'positionIdx': position_idx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })

        current_position_type = signal
        print(f"✅ {signal.upper()} position placed.")

    except Exception as e:
        print(f"[Order Error]: {str(e)}")


def run_bot():
    global current_position_type

    print(f"\nRunning bot at {datetime.datetime.now()}")
    try:
        set_leverage(symbol, leverage)

        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_ema(df, ema_short_period, ema_long_period)
        df = calculate_atr(df)

        signal = get_ema_signal(df)
        if not signal:
            print("No crossover signal.")
            return

        # ATR filter
        average_price = df['close'].iloc[-14:].mean()
        dynamic_atr_threshold = average_price * 0.001
        if df['atr'].iloc[-1] < dynamic_atr_threshold:
            print(f"Low ATR ({df['atr'].iloc[-1]:.6f}) — skipping.")
            return

        # EMA50 trend filter (optional)
        if enable_ema50_filter:
            price = df['close'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            if signal == 'buy' and price < ema_50:
                print("Skipping long (price < EMA50)")
                return
            elif signal == 'sell' and price > ema_50:
                print("Skipping short (price > EMA50)")
                return

        if current_position_type is None:
            print(f"First signal: {signal.upper()}")
            place_trade(signal, df)

        elif signal != current_position_type:
            print(f"Opposite signal: closing {current_position_type.upper()} and entering {signal.upper()}")
            close_position()
            time.sleep(1)
            place_trade(signal, df)

        else:
            print(f"Signal {signal.upper()} already open — no action.")

    except Exception as e:
        print(f"[Bot Error]: {str(e)}")


# ====== MAIN LOOP ======
while True:
    run_bot()
    time.sleep(60)
