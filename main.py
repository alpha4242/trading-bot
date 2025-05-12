import ccxt
import pandas as pd
import time
import datetime
import threading
import os

# ====== CONFIGURATION ======
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

symbol = 'PNUTUSDT'
timeframe = '1m'
ema_short_period = 9
ema_long_period = 21
quantity = 15  # Total position size
leverage = 10
stoploss_lookback = 4  # For SL calculation
rsi_diff_threshold = 8
enable_ema50_filter = False
enable_rsi_exit = False

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
last_signal = None

# ====== ENHANCED POSITION MANAGEMENT ======
def force_close_all_positions():
    """Nuclear option to close ALL positions for the symbol"""
    try:
        # 1. Cancel all active orders first
        exchange.cancel_all_orders(symbol)
        
        # 2. Get all open positions
        positions = exchange.fetch_positions([symbol])
        
        # 3. Close all positions
        for pos in positions:
            if float(pos['contracts']) > 0:
                close_side = 'sell' if pos['side'].lower() == 'long' else 'buy'
                exchange.create_order(
                    symbol,
                    'market',
                    close_side,
                    float(pos['contracts']),
                    None,
                    {'reduceOnly': True, 'positionIdx': int(pos['info']['positionIdx'])}
                )
                print(f"🔴 Force-closing {pos['side']} position")
        
        # 4. Verify closure
        for _ in range(5):
            time.sleep(1)
            remaining_positions = [p for p in exchange.fetch_positions([symbol]) 
                                if float(p['contracts']) > 0]
            if not remaining_positions:
                print("✅ All positions confirmed closed")
                return True
                
        print("❗ Some positions may remain open after forced closure")
        return False
        
    except Exception as e:
        print(f"💥 Emergency close failed: {str(e)}")
        return False

def close_position(positionIdx):
    """Enhanced position closing with verification"""
    for attempt in range(3):  # 3 attempts
        try:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if 
                      int(p['info']['positionIdx']) == positionIdx and 
                      float(p['contracts']) > 0), None)
            
            if not pos:
                return True  # Already closed
                
            # Market order to close
            close_side = 'sell' if positionIdx == 1 else 'buy'
            exchange.create_order(
                symbol,
                'market',
                close_side,
                float(pos['contracts']),
                None,
                {'reduceOnly': True, 'positionIdx': positionIdx}
            )
            
            # Verification
            for _ in range(5):
                time.sleep(1)
                current_pos = exchange.fetch_positions([symbol])
                if not any(p for p in current_pos if 
                          int(p['info']['positionIdx']) == positionIdx and 
                          float(p['contracts']) > 0):
                    return True
                    
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)
    
    print("🔴 Falling back to force-close-all")
    return force_close_all_positions()

def sync_position_state():
    """Synchronize internal state with actual positions"""
    global is_long_open, is_short_open
    try:
        positions = exchange.fetch_positions([symbol])
        is_long_open = any(p for p in positions if int(p['info']['positionIdx']) == 1 and float(p['contracts']) > 0)
        is_short_open = any(p for p in positions if int(p['info']['positionIdx']) == 2 and float(p['contracts']) > 0)
    except Exception as e:
        print(f"[State Sync Error]: {str(e)}")

# ====== TRADING FUNCTIONS ======
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
    print(f"[MONITOR] Started monitoring {position_type.upper()} position...")

    while True:
        time.sleep(15)
        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_ema(df, ema_short_period, ema_long_period)
        df = calculate_rsi(df)

        if enable_rsi_exit:
            rsi = df['rsi'].iloc[-1]
            rsi_sma = df['rsi_sma'].iloc[-1]
            diff = abs(rsi - rsi_sma)

            if position_type == 'buy' and rsi < rsi_sma and diff >= rsi_diff_threshold:
                print("[EXIT] RSI exit condition met for BUY")
                close_position(1)
                is_long_open = False
                return
            elif position_type == 'sell' and rsi > rsi_sma and diff >= rsi_diff_threshold:
                print("[EXIT] RSI exit condition met for SELL")
                close_position(2)
                is_short_open = False
                return

def place_order(signal, df):
    global is_long_open, is_short_open, last_signal
    
    # Sync with actual positions before trading
    sync_position_state()
    
    recent_candles = df[-stoploss_lookback:]
    current_price = df['close'].iloc[-1]

    if signal == last_signal:
        return

    # 1. Close opposite position first
    if (signal == 'buy' and is_short_open) or (signal == 'sell' and is_long_open):
        position_to_close = 2 if is_short_open else 1
        print(f"[REVERSE] {signal.upper()} signal during existing position. Closing first...")
        
        if not close_position(position_to_close):
            print("❌ Critical: Failed to close position - aborting trade")
            return
        
        # Update state only after successful closure
        is_long_open = False
        is_short_open = False
        time.sleep(2)  # Mandatory cooling period
        sync_position_state()  # Double-check

    # 2. Calculate risk and position splits
    if signal == 'buy':
        sl_price = recent_candles['low'].min()
        risk = current_price - sl_price
        tp_prices = [
            current_price + 2 * risk,  # 1:2
            current_price + 3 * risk,  # 1:3
            current_price + 4 * risk   # 1:4
        ]
    else:  # sell
        sl_price = recent_candles['high'].max()
        risk = sl_price - current_price
        tp_prices = [
            current_price - 2 * risk,  # 1:2
            current_price - 3 * risk,  # 1:3
            current_price - 4 * risk   # 1:4
        ]

    # 3. Position sizing (40% + 30% + 20% + 10% runner)
    quantities = [
        round(quantity * 0.40, 3),  # 40% for 1:2
        round(quantity * 0.30, 3),  # 30% for 1:3
        round(quantity * 0.20, 3),  # 20% for 1:4
        round(quantity * 0.10, 3)   # 10% runner
    ]

    # 4. Execute orders
    try:
        positionIdx = 1 if signal == 'buy' else 2
        close_side = 'sell' if signal == 'buy' else 'buy'
        
        # TP1 (1:2)
        exchange.create_order(symbol, 'market', signal, quantities[0], None, {
            'positionIdx': positionIdx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })
        exchange.create_order(symbol, 'limit', close_side, quantities[0], round(tp_prices[0], 4), {
            'positionIdx': positionIdx,
            'reduceOnly': True
        })

        # TP2 (1:3)
        exchange.create_order(symbol, 'market', signal, quantities[1], None, {
            'positionIdx': positionIdx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })
        exchange.create_order(symbol, 'limit', close_side, quantities[1], round(tp_prices[1], 4), {
            'positionIdx': positionIdx,
            'reduceOnly': True
        })

        # TP3 (1:4)
        exchange.create_order(symbol, 'market', signal, quantities[2], None, {
            'positionIdx': positionIdx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })
        exchange.create_order(symbol, 'limit', close_side, quantities[2], round(tp_prices[2], 4), {
            'positionIdx': positionIdx,
            'reduceOnly': True
        })

        # Runner (10%)
        exchange.create_order(symbol, 'market', signal, quantities[3], None, {
            'positionIdx': positionIdx,
            'stopLoss': round(sl_price, 4),
            'slTriggerBy': 'LastPrice'
        })

        # Update state
        if signal == 'buy':
            is_long_open = True
        else:
            is_short_open = True
        last_signal = signal
        
        print(f"✅ Executed {signal.upper()} order with 3TPs+Runner")
        print(f"TP1: {round(tp_prices[0], 4)} (1:2, {quantities[0]} contracts)")
        print(f"TP2: {round(tp_prices[1], 4)} (1:3, {quantities[1]} contracts)")
        print(f"TP3: {round(tp_prices[2], 4)} (1:4, {quantities[2]} contracts)") 
        print(f"Runner: No TP ({quantities[3]} contracts)")
        
        threading.Thread(target=monitor_position, args=(signal,)).start()

    except Exception as e:
        print(f"[Order Error]: {str(e)}")
        force_close_all_positions()

def run_bot():
    print(f"\nRunning bot at {datetime.datetime.now()}")
    try:
        set_leverage(symbol, leverage)
        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_ema(df, ema_short_period, ema_long_period)
        df = calculate_rsi(df)
        df = calculate_atr(df)

        signal = get_ema_signal(df)
        if not signal:
            print("ANALYSING THE MARKET")
            return

        if enable_ema50_filter:
            price = df['close'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            if signal == 'buy' and price < ema_50:
                print("Skipping long because market is bearish (price < EMA50)")
                return
            elif signal == 'sell' and price > ema_50:
                print("Skipping short because market is bullish (price > EMA50)")
                return

        average_price = df['close'].iloc[-14:].mean()
        dynamic_atr_threshold = average_price * 0.001
        if df['atr'].iloc[-1] < dynamic_atr_threshold:
            print(f"Low ATR ({df['atr'].iloc[-1]:.6f}) below threshold ({dynamic_atr_threshold:.6f}) - skipping.")
            return

        print(f"Confirmed signal: {signal.upper()}")
        place_order(signal, df)

    except Exception as e:
        print(f"[Bot Error]: {str(e)}")
        force_close_all_positions()

# ====== MAIN LOOP ======
if __name__ == "__main__":
    print("=== Starting Trading Bot ===")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Strategy: EMA{ema_short_period}/EMA{ema_long_period} Crossover")
    print(f"Risk Management: 3TPs (1:2,1:3,1:4) + 10% Runner")
    
    # Initial position sync
    sync_position_state()
    
    while True:
        try:
            run_bot()
        except Exception as e:
            print(f"💣 Main loop crash: {str(e)}")
            force_close_all_positions()
        time.sleep(60)
