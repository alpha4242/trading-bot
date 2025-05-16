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
timeframe = '5m'
ema_short_period = 9
ema_long_period = 21
quantity = 6  # Position size per trade
leverage = 10
enable_adx_filter = False
adx_threshold = 20
adx_period = 14

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
trade_history = []
current_trend = None  # 'bullish' or 'bearish'
active_trailing_stops = {}

# ====== TECHNICAL INDICATORS ======
def calculate_ema(df):
    df['ema_9'] = df['close'].ewm(span=ema_short_period, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=ema_long_period, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema_150'] = df['close'].ewm(span=150, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()
    return df

def check_ema_stack(df):
    global current_trend
    
    ema_50 = df['ema_50'].iloc[-1]
    ema_100 = df['ema_100'].iloc[-1]
    ema_150 = df['ema_150'].iloc[-1]
    ema_200 = df['ema_200'].iloc[-1]
    
    # Check for bearish stack (50 < 100 < 150 < 200)
    if ema_50 < ema_100 < ema_150 < ema_200:
        current_trend = 'bearish'
        return 'bearish'
    
    # Check for bullish stack (50 > 100 > 150 > 200)
    elif ema_50 > ema_100 > ema_150 > ema_200:
        current_trend = 'bullish'
        return 'bullish'
    
    # Check if trend is broken (50 crosses 100)
    elif current_trend == 'bearish' and ema_50 > ema_100:
        current_trend = None
        return 'trend_broken'
    elif current_trend == 'bullish' and ema_50 < ema_100:
        current_trend = None
        return 'trend_broken'
    
    return None

def get_ema_signal(df):
    ema_short_prev2 = df['ema_9'].iloc[-3]
    ema_long_prev2 = df['ema_21'].iloc[-3]
    ema_short_prev1 = df['ema_9'].iloc[-2]
    ema_long_prev1 = df['ema_21'].iloc[-2]
    
    if current_trend == 'bearish':
        if ema_short_prev2 > ema_long_prev2 and ema_short_prev1 < ema_long_prev1:
            return 'sell'
    elif current_trend == 'bullish':
        if ema_short_prev2 < ema_long_prev2 and ema_short_prev1 > ema_long_prev1:
            return 'buy'
    return None

# ====== POSITION MANAGEMENT ======
def force_close_all_positions():
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
    global is_long_open, is_short_open
    try:
        positions = exchange.fetch_positions([symbol])
        is_long_open = any(p for p in positions if int(p['info']['positionIdx']) == 1 and float(p['contracts']) > 0)
        is_short_open = any(p for p in positions if int(p['info']['positionIdx']) == 2 and float(p['contracts']) > 0)
    except Exception as e:
        print(f"[State Sync Error]: {str(e)}")

def manage_trailing_stop(position_type, entry_price):
    positionIdx = 1 if position_type == 'long' else 2
    trade_id = f"{position_type}_{time.time()}"
    active_trailing_stops[trade_id] = True
    
    print(f"\n=== Trailing Stop Active ({position_type.upper()}) ===")
    
    while active_trailing_stops.get(trade_id, False):
        try:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if 
                       int(p['info']['positionIdx']) == positionIdx and 
                       float(p['contracts']) > 0), None)
            
            if not pos:
                print("Position closed - Stopping trailing stop")
                break
                
            df = fetch_ohlcv(symbol, timeframe, limit=200)
            df = calculate_ema(df)
            current_200ema = df['ema_200'].iloc[-1]
            current_price = float(pos['markPrice'])
            
            if position_type == 'long':
                risk = entry_price - current_200ema
                reward = current_price - entry_price
                
                if reward >= risk:  # Only trail after 1:1
                    new_sl = current_200ema
                    current_sl = float(pos['stopLoss'] or 0)
                    
                    if new_sl > current_sl:
                        try:
                            exchange.create_order(
                                symbol,
                                'market',
                                'sell',
                                0,
                                None,
                                {
                                    'positionIdx': positionIdx,
                                    'stopLoss': round(new_sl, 4),
                                    'slTriggerBy': 'LastPrice',
                                    'reduceOnly': True
                                }
                            )
                            print(f"Updated trailing stop to {new_sl:.4f}")
                        except Exception as e:
                            print(f"Trailing stop update failed: {str(e)}")
            else:  # short
                risk = current_200ema - entry_price
                reward = entry_price - current_price
                
                if reward >= risk:  # Only trail after 1:1
                    new_sl = current_200ema
                    current_sl = float(pos['stopLoss'] or float('inf'))
                    
                    if new_sl < current_sl:
                        try:
                            exchange.create_order(
                                symbol,
                                'market',
                                'buy',
                                0,
                                None,
                                {
                                    'positionIdx': positionIdx,
                                    'stopLoss': round(new_sl, 4),
                                    'slTriggerBy': 'LastPrice',
                                    'reduceOnly': True
                                }
                            )
                            print(f"Updated trailing stop to {new_sl:.4f}")
                        except Exception as e:
                            print(f"Trailing stop update failed: {str(e)}")
            
            time.sleep(15)
            
        except Exception as e:
            print(f"[Trailing Stop Error]: {str(e)}")
            time.sleep(30)
    
    active_trailing_stops.pop(trade_id, None)

def log_pnl(position_type):
    print(f"\n=== PNL Monitoring ({position_type.upper()}) ===")
    positionIdx = 1 if position_type == 'long' else 2
    
    while True:
        try:
            positions = exchange.fetch_positions([symbol])
            pos = next((p for p in positions if 
                       int(p['info']['positionIdx']) == positionIdx and 
                       float(p['contracts']) > 0), None)
            
            if not pos:
                break
                
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                f"Unrealized PNL: ${float(pos['unrealizedPnl']):.2f} | "
                f"Entry: {float(pos['entryPrice']):.4f} | "
                f"SL: {float(pos['stopLoss'] or 0):.4f}"
            )
            time.sleep(15)
        except Exception as e:
            print(f"[PNL Error]: {str(e)}")
            break

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

def place_order(signal, df):
    global is_long_open, is_short_open, last_signal, trade_history
    
    sync_position_state()
    current_price = df['close'].iloc[-1]
    
    # Only close opposite positions
    if (signal == 'buy' and is_short_open) or (signal == 'sell' and is_long_open):
        position_to_close = 2 if is_short_open else 1
        if not close_position(position_to_close):
            return
        is_long_open = is_short_open = False
        time.sleep(2)
        sync_position_state()

    # Set initial stoploss at 200 EMA
    sl_price = df['ema_200'].iloc[-1]
    
    # Calculate risk parameters
    if signal == 'buy':
        risk = current_price - sl_price
        tp_prices = [current_price + r * risk for r in [1, 2, 3]]  # 1:1, 1:2, 1:3
    else:
        risk = sl_price - current_price
        tp_prices = [current_price - r * risk for r in [1, 2, 3]]

    try:
        positionIdx = 1 if signal == 'buy' else 2
        close_side = 'sell' if signal == 'buy' else 'buy'
        
        # Execute market order
        exchange.create_order(
            symbol,
            'market',
            signal,
            quantity,
            None,
            {
                'positionIdx': positionIdx,
                'stopLoss': round(sl_price, 4),
                'slTriggerBy': 'LastPrice'
            }
        )
        
        # Set take profits (split quantity equally)
        tp_qty = round(quantity / len(tp_prices), 3)
        for i, tp_price in enumerate(tp_prices):
            exchange.create_order(
                symbol,
                'limit',
                close_side,
                tp_qty,
                round(tp_price, 4),
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
            'trend': current_trend
        })
        
        print(f"\n✅ Executed {signal.upper()} Order (Trend: {current_trend})")
        print(f"Entry: {current_price:.4f} | SL (200EMA): {sl_price:.4f}")
        print("Take Profits:")
        for i, tp in enumerate(tp_prices):
            print(f"  TP{i+1}: {tp:.4f} (1:{i+1})")
        
        # Start monitoring threads
        threading.Thread(target=manage_trailing_stop, args=(signal, current_price)).start()
        threading.Thread(target=log_pnl, args=(signal,)).start()

    except Exception as e:
        print(f"[Order Error]: {str(e)}")
        force_close_all_positions()

def run_bot():
    print(f"\nRunning bot at {datetime.datetime.now()}")
    try:
        set_leverage(symbol, leverage)
        df = fetch_ohlcv(symbol, timeframe)
        df = calculate_ema(df)
        df = calculate_rsi(df)
        df = calculate_atr(df)
        
        # Check EMA stack for trend
        trend_status = check_ema_stack(df)
        
        if trend_status == 'trend_broken':
            print("❗ Trend broken (50EMA crossed 100EMA) - Closing all positions")
            force_close_all_positions()
            return
        
        if not current_trend:
            print("No clear trend (EMAs not stacked properly)")
            return
            
        print(f"\nCurrent Trend: {current_trend.upper()}")
        print(f"50EMA: {df['ema_50'].iloc[-1]:.4f}")
        print(f"100EMA: {df['ema_100'].iloc[-1]:.4f}")
        print(f"150EMA: {df['ema_150'].iloc[-1]:.4f}")
        print(f"200EMA: {df['ema_200'].iloc[-1]:.4f}")

        signal = get_ema_signal(df)
        if not signal:
            print("No valid signal - Waiting")
            return
            
        # Additional filters
        avg_price = df['close'].iloc[-14:].mean()
        if df['atr'].iloc[-1] < avg_price * 0.001:
            print("Low volatility (ATR filter) - Skipping trade")
            return

        print(f"Confirmed {signal.upper()} signal in {current_trend} trend")
        place_order(signal, df)

    except Exception as e:
        print(f"[Bot Error]: {str(e)}")
        force_close_all_positions()

# ====== MAIN LOOP ======
if __name__ == "__main__":
    print("=== EMA Stack Trend-Following Bot ===")
    print(f"Symbol: {symbol} | TF: {timeframe}")
    print(f"Strategy: EMA9/21 Cross in EMA50/100/150/200 Trend")
    print(f"Risk: 3TPs (1:1,1:2,1:3) + Trailing 200EMA after 1:1")
    
    sync_position_state()
    
    while True:
        try:
            run_bot()
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            # Stop all trailing threads
            for trade_id in list(active_trailing_stops.keys()):
                active_trailing_stops[trade_id] = False
            force_close_all_positions()
            break
        except Exception as e:
            print(f"💣 Critical error: {str(e)}")
            force_close_all_positions()
            time.sleep(60)
