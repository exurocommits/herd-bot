"""
Pure Python utility functions for indicator calculations.
No numpy dependency — works with plain Python lists.
"""
import math
from typing import List, Optional, Tuple


def sma(data: List[float], period: int) -> List[float]:
    """Simple Moving Average. Returns list same length as data (NaN-padded)."""
    result = [float('nan')] * len(data)
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        result[i] = sum(window) / period
    return result


def ema(data: List[float], period: int) -> List[float]:
    """Exponential Moving Average."""
    result = [float('nan')] * len(data)
    if len(data) < period:
        return result
    
    # Seed with SMA
    seed = sum(data[:period]) / period
    result[period - 1] = seed
    k = 2.0 / (period + 1)
    
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i - 1] * (1 - k)
    
    return result


def rsi(data: List[float], period: int = 14) -> List[float]:
    """Relative Strength Index."""
    result = [float('nan')] * len(data)
    if len(data) < period + 1:
        return result
    
    gains = []
    losses = []
    for i in range(1, len(data)):
        diff = data[i] - data[i - 1]
        gains.append(max(0.0, diff))
        losses.append(max(0.0, -diff))
    
    # Initial average gain/loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
    """Average True Range."""
    result = [float('nan')] * len(high)
    if len(high) < 2:
        return result
    
    # True range
    tr = [0.0] * len(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # ATR via EMA-style smoothing
    if len(tr) < period:
        for i in range(len(tr)):
            result[i] = tr[i]
        return result
    
    result[period - 1] = sum(tr[:period]) / period
    for i in range(period, len(tr)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    
    return result


def bollinger_bands(data: List[float], period: int = 20, num_std: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """Returns (upper, middle, lower) bands."""
    middle = sma(data, period)
    upper = [float('nan')] * len(data)
    lower = [float('nan')] * len(data)
    
    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    
    return upper, middle, lower


def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """MACD line, signal line, histogram."""
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)
    
    macd_line = [float('nan')] * len(data)
    for i in range(len(data)):
        if not math.isnan(fast_ema[i]) and not math.isnan(slow_ema[i]):
            macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # Signal line = EMA of MACD line
    valid_macd = [x if not math.isnan(x) else 0.0 for x in macd_line]
    signal_line = ema(valid_macd, signal)
    
    histogram = [float('nan')] * len(data)
    for i in range(len(data)):
        if not math.isnan(macd_line[i]) and not math.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


def stochastic(high: List[float], low: List[float], close: List[float], 
               k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
    """Stochastic Oscillator. Returns (%K, %D)."""
    k_line = [float('nan')] * len(close)
    
    for i in range(k_period - 1, len(close)):
        highest = max(high[i - k_period + 1:i + 1])
        lowest = min(low[i - k_period + 1:i + 1])
        if highest == lowest:
            k_line[i] = 50.0
        else:
            k_line[i] = ((close[i] - lowest) / (highest - lowest)) * 100.0
    
    d_line = sma(k_line, d_period)
    return k_line, d_line


def supertrend(high: List[float], low: List[float], close: List[float],
               period: int = 10, multiplier: float = 3.0) -> Tuple[List[float], List[int]]:
    """Supertrend indicator. Returns (supertrend_values, directions: 1=up, -1=down)."""
    atr_vals = atr(high, low, close, period)
    st = [float('nan')] * len(close)
    direction = [0] * len(close)
    
    if len(close) < period:
        return st, direction
    
    # Initialize
    hl2 = [(h + l) / 2 for h, l in zip(high, low)]
    
    upper_band = [0.0] * len(close)
    lower_band = [0.0] * len(close)
    
    for i in range(period - 1, len(close)):
        if math.isnan(atr_vals[i]):
            continue
        
        ub = hl2[i] + multiplier * atr_vals[i]
        lb = hl2[i] - multiplier * atr_vals[i]
        
        # Upper band: can only go down (or stay) when price is above it
        if i > period - 1 and not math.isnan(upper_band[i - 1]):
            if ub < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                upper_band[i] = ub
            else:
                upper_band[i] = upper_band[i - 1]
        else:
            upper_band[i] = ub
        
        # Lower band: can only go up (or stay) when price is below it
        if i > period - 1 and not math.isnan(lower_band[i - 1]):
            if lb > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                lower_band[i] = lb
            else:
                lower_band[i] = lower_band[i - 1]
        else:
            lower_band[i] = lb
        
        # Direction
        if i > period - 1:
            if close[i] > upper_band[i - 1]:
                direction[i] = 1
            elif close[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1] if direction[i - 1] != 0 else 1
        else:
            direction[i] = 1
        
        st[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    return st, direction


def adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
    """Average Directional Index."""
    result = [float('nan')] * len(close)
    if len(close) < period + 1:
        return result
    
    # Calculate +DM, -DM, TR
    plus_dm = [0.0] * len(close)
    minus_dm = [0.0] * len(close)
    tr = [0.0] * len(close)
    
    for i in range(1, len(close)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth with Wilder's method
    smooth_tr = [0.0] * len(close)
    smooth_plus_dm = [0.0] * len(close)
    smooth_minus_dm = [0.0] * len(close)
    
    smooth_tr[period] = sum(tr[1:period + 1])
    smooth_plus_dm[period] = sum(plus_dm[1:period + 1])
    smooth_minus_dm[period] = sum(minus_dm[1:period + 1])
    
    for i in range(period + 1, len(close)):
        smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
        smooth_plus_dm[i] = smooth_plus_dm[i - 1] - (smooth_plus_dm[i - 1] / period) + plus_dm[i]
        smooth_minus_dm[i] = smooth_minus_dm[i - 1] - (smooth_minus_dm[i - 1] / period) + minus_dm[i]
    
    # +DI, -DI
    plus_di = [0.0] * len(close)
    minus_di = [0.0] * len(close)
    dx = [0.0] * len(close)
    
    for i in range(period, len(close)):
        if smooth_tr[i] > 0:
            plus_di[i] = 100.0 * smooth_plus_dm[i] / smooth_tr[i]
            minus_di[i] = 100.0 * smooth_minus_dm[i] / smooth_tr[i]
            total = plus_di[i] + minus_di[i]
            if total > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / total
    
    # ADX = smoothed DX
    result[2 * period - 1] = sum(dx[period:2 * period]) / period
    for i in range(2 * period, len(close)):
        result[i] = (result[i - 1] * (period - 1) + dx[i]) / period
    
    return result


def cross_above(series_a: List[float], series_b: List[float], idx: int) -> bool:
    """Check if series_a crossed above series_b at index idx."""
    if idx < 1:
        return False
    a_curr, a_prev = series_a[idx], series_a[idx - 1]
    b_curr, b_prev = series_b[idx], series_b[idx - 1]
    
    if any(math.isnan(x) for x in [a_curr, a_prev, b_curr, b_prev]):
        return False
    return a_prev <= b_prev and a_curr > b_curr


def cross_below(series_a: List[float], series_b: List[float], idx: int) -> bool:
    """Check if series_a crossed below series_b at index idx."""
    if idx < 1:
        return False
    a_curr, a_prev = series_a[idx], series_a[idx - 1]
    b_curr, b_prev = series_b[idx], series_b[idx - 1]
    
    if any(math.isnan(x) for x in [a_curr, a_prev, b_curr, b_prev]):
        return False
    return a_prev >= b_prev and a_curr < b_curr
