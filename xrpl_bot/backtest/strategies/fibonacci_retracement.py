import numpy as np
import pandas as pd

class FibonacciRetracementStrategy:
    """
    Fibonacci Retracement Strategy.
    
    Track last swing high/low (20-bar lookback).
    Calculate fib levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%.
    BUY at 61.8% retracement level when price bounces (close > low of bar that touched 61.8%).
    SELL at 0% or 23.6% level.
    Confidence: bounce strength / ATR.
    """
    def __init__(self, data):
        self.df = data
        self.lookback = 20
        self.atr_period = 14
        self._calculate_indicators()

    def _calculate_indicators(self):
        # ATR for confidence
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['atr'] = true_range.rolling(window=self.atr_period).mean()

    def next(self, bar):
        if bar < self.lookback or bar >= len(self.df):
            return {"type": "HOLD", "price": self.df['close'].iloc[bar] if bar < len(self.df) else 0.0, "confidence": 0.0, "metadata": {}}

        window = self.df.iloc[bar - self.lookback : bar]
        swing_high = window['high'].max()
        swing_low = window['low'].min()
        
        diff = swing_high - swing_low
        if diff == 0:
            return {"type": "HOLD", "price": float(self.df['close'].iloc[bar]), "confidence": 0.0, "metadata": {}}

        # Levels
        levels = {
            "0.0": swing_high,
            "0.236": swing_high - 0.236 * diff,
            "0.382": swing_high - 0.382 * diff,
            "0.5": swing_high - 0.5 * diff,
            "0.618": swing_high - 0.618 * diff,
            "0.786": swing_high - 0.786 * diff,
            "1.0": swing_low
        }

        current_close = self.df['close'].iloc[bar]
        current_low = self.df['low'].iloc[bar]
        atr = self.df['atr'].iloc[bar]
        
        signal_type = "HOLD"
        confidence = 0.0
        metadata = {"levels": levels}

        # Check for 61.8% bounce
        # We look for a bar that touched/went below 61.8% and now closes above it
        fib_618 = levels["0.618"]
        
        # Logic: if the current bar's low <= 61.8% AND current close > 61.8% (a bounce)
        # Or if the previous bar was the touch
        if current_low <= fib_618 and current_close > fib_618:
            bounce_strength = current_close - current_low
            confidence = (bounce_strength / atr) if atr > 0 else 0.0
            signal_type = "BUY"
            metadata["bounce_strength"] = float(bounce_strength)
        
        # Sell logic: close at 0% (swing high) or 23.6%
        elif current_close >= levels["0.0"] or current_close <= levels["0.236"]:
             signal_type = "SELL"
             confidence = 0.5
             metadata["exit_level"] = "0 or 23.6"

        return {
            "type": signal_type,
            "price": float(current_close),
            "confidence": float(confidence),
            "metadata": metadata
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"lookback": self.lookback, "atr_period": self.atr_period}
