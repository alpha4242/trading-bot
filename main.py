import ccxt
import pandas as pd
import numpy as np
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
enable_adx_filter = True  # ADX control switch
adx_threshold = 20  # Minimum ADX value for valid trend
adx_period = 14  # ADX calculation period

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
trade_history = []  # For tracking performance

# ====== TECHNICAL INDICATORS ======
def calculate_adx(df, period=14):
    """Calculate ADX with +DI and -DI"""
    # Calculate True Range
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smoothing (Wilder's method)
    df['atr'] = df['tr'].rolling(window=period).mean()
    df['+di'] = 100 * (df['+dm'].rolling(window=period).mean() / df['atr'])
    df['-di'] = 100 * (df['-dm'].rolling(window=period).mean() / df['atr'])
    
    # Calculate DX and ADX
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    return df.drop(['tr0', 'tr1', 'tr2', 'up_move', 'down_move'], axis=1)

# ====== POSITION MANAGEMENT ======
def force_close_all_positions():
    """Nuclear option to close ALL positions for the symbol"""
    try:
        exchange.cancel_all_orders(symbol)
        positions = exchange.fetch_positions([symbol])
        
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
        
        for _ in range(5):
            time.sleep(1)
            if not any(p for p in exchange.fetch_positions([symbol]) if float(p['contracts']) > 0):
                print("✅ All positions confirmed closed")
                return True
                
        print("❗ Some positions may remain open")
        return False
        
    except Exception as e:
        print(f"💥 Emergency close failed: {str(e)}")
        return False

def close_position(positionIdx):
    """Enhanced position closing with verification"""
    for attempt in range(3):
        try:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if 
                      int(p['info']['positionIdx']) == positionIdx and 
                      float(p['contracts']) > 0), None)
            
            if not pos:
                return True
                
            close_side = 'sell' if positionIdx == 1 else 'buy'
            exchange.create_order(
                symbol,
                'market',
                close_side,
                float(pos['contracts']),
                None,
                {'reduceOnly': True, 'positionIdx': positionIdx}
            )
            
            for _ in range(5):
                time.sleep(1)
                if not any(p for p in exchange.fetch_positions([symbol]) 
                          if int(p['info']['positionIdx']) == positionIdx and 
                          float(p['contracts']) > 0):
                    return True
                    
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)
    
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
    try:
        exchange.set_leverage(leverage, symbol)
        print(f"Leverage set to {leverage}x")
    except ccxt.BaseError as e:
        if "leverage not modified" not in str(e).lower():
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
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    df['rsi'] = 100 - (100 / (1 + gain.rolling(period).mean() / loss.rolling(period).mean()))
    df['rsi_sma'] = df['rsi'].rolling(period).mean()
    return df

def calculate_atr(df, period=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['atr'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1).rolling(period).mean()
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

def log_pnl(position_type):
    """Real-time PNL monitoring"""
    print(f"\n=== PNL Monitoring ({position_type.upper()}) ===")
    while True:
        try:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if 
                       p['symbol'] == symbol and 
                       float(p['contracts']) > 0), None)
            
            if not pos:
                break
                
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                f"Unrealized PNL: ${float(pos['unrealizedPnl']):.2f} | "
                f"Entry: {float(pos['entryPrice']):.4f}"
            )
            time.sleep(15)
        except Exception as e:
            print(f"[PNL Error]: {str(e)}")
            break

def place_order(signal, df):
    global is_long_open, is_short_open, last_signal, trade_history
    
    sync_position_state()
    recent_candles = df[-stoploss_lookback:]
    current_price = df['close'].iloc[-1]

    if signal == last_signal:
        return

    # Close opposite position
    if (signal == 'buy' and is_short_open) or (signal == 'sell' and is_long_open):
        position_to_close = 2 if is_short_open else 1
        if not close_position(position_to_close):
            return
        is_long_open = is_short_open = False
        time.sleep(2)
        sync_position_state()

    # Calculate risk parameters
    if signal == 'buy':
        sl_price = recent_candles['low'].min()
        risk = current_price - sl_price
        tp_prices = [current_price + r * risk for r in [2, 3, 4]]  # 1:2, 1:3, 1:4
    else:
        sl_price = recent_candles['high'].max()
        risk = sl_price - current_price
        tp_prices = [current_price - r * risk for r in [2, 3, 4]]

    quantities = [
        round(quantity * p, 3) for p in [0.40, 0.30, 0.20]  # 40%, 30%, 20%
    ] + [round(quantity * 0.10, 3)]  # 10% runner

    try:
        positionIdx = 1 if signal == 'buy' else 2
        close_side = 'sell' if signal == 'buy' else 'buy'
        
        # Execute orders for all 4 portions
        for i in range(4):
            exchange.create_order(
                symbol,
                'market',
                signal,
                quantities[i],
                None,
                {
                    'positionIdx': positionIdx,
                    'stopLoss': round(sl_price, 4),
                    'slTriggerBy': 'LastPrice'
                }
            )
            if i < 3:  # Add TP for first 3 portions
                exchange.create_order(
                    symbol,
                    'limit',
                    close_side,
                    quantities[i],
                    round(tp_prices[i], 4),
                    {
                        'positionIdx': positionIdx,
                        'reduceOnly': True
                    }
                )

        # Update state and log
        if signal == 'buy':
            is_long_open = True
        else:
            is_short_open = True
        last_signal = signal
        
        trade_history.append({
            'time': datetime.datetime.now(),
            'signal': signal,
            'entry': current_price,
            'size': quantity,
            'sl': sl_price,
            'tps': tp_prices,
            'adx': df['adx'].iloc[-1],
            '+di': df['+di'].iloc[-1],
            '-di': df['-di'].iloc[-1]
        })
        
        print(f"\n✅ Executed {signal.upper()} Order")
        print(f"Entry: {current_price:.4f} | SL: {sl_price:.4f}")
        print(f"ADX: {df['adx'].iloc[-1]:.2f} (+DI: {df['+di'].iloc[-1]:.2f}, -DI: {df['-di'].iloc[-1]:.2f})")
        print("Take Profits:")
        print(f"  TP1: {tp_prices[0]:.4f} (1:2, {quantities[0]} contracts)")
        print(f"  TP2: {tp_prices[1]:.4f} (1:3, {quantities[1]} contracts)")
        print(f"  TP3: {tp_prices[2]:.4f} (1:4, {quantities[2]} contracts)")
        print(f"  Runner: {quantities[3]} contracts (No TP)")
        
        threading.Thread(target=log_pnl, args=(signal,)).start()

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
        df = calculate_adx(df, adx_period)

        # Print ADX values in logs
        current_adx = df['adx'].iloc[-1]
        plus_di = df['+di'].iloc[-1]
        minus_di = df['-di'].iloc[-1]
        
        print(f"\nCurrent ADX: {current_adx:.2f}")
        print(f"+DI: {plus_di:.2f} | -DI: {minus_di:.2f}")
        print(f"Trend Strength: {'Strong' if current_adx >= adx_threshold else 'Weak'}")

        signal = get_ema_signal(df)
        if not signal:
            print("No crossover signal")
            return

        # ADX Filter
        if enable_adx_filter:
            if current_adx < adx_threshold:
                print(f"ADX {current_adx:.2f} < threshold {adx_threshold} - Skipping trade")
                return
                
            if (signal == 'buy' and plus_di < minus_di) or (signal == 'sell' and minus_di < plus_di):
                print("Directional momentum weak, skipping trade")
                return

        # Other filters
        if enable_ema50_filter and (
            (signal == 'buy' and df['close'].iloc[-1] < df['ema_50'].iloc[-1]) or
            (signal == 'sell' and df['close'].iloc[-1] > df['ema_50'].iloc[-1])
        ):
            print("EMA50 filter triggered")
            return

        avg_price = df['close'].iloc[-14:].mean()
        if df['atr'].iloc[-1] < avg_price * 0.001:
            print("Low volatility (ATR filter)")
            return

        print(f"Confirmed {signal.upper()} signal with ADX {current_adx:.2f}")
        place_order(signal, df)

    except Exception as e:
        print(f"[Bot Error]: {str(e)}")
        force_close_all_positions()

# ====== MAIN LOOP ======
if __name__ == "__main__":
    print("=== Trend-Following Bot ===")
    print(f"Symbol: {symbol} | TF: {timeframe}")
    print(f"Strategy: EMA{ema_short_period}/{ema_long_period} Crossover")
    print(f"Risk: 3TPs (1:2,1:3,1:4) + 10% Runner")
    print(f"ADX Filter: {'ON' if enable_adx_filter else 'OFF'} (Threshold: {adx_threshold})")
    
    sync_position_state()
    
    while True:
        try:
            run_bot()
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            force_close_all_positions()
            break
        except Exception as e:
            print(f"💣 Critical error: {str(e)}")
            force_close_all_positions()
            time.sleep(60)
