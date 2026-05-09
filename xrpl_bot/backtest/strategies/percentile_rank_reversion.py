import numpy as np
import pandas as pd

class PercentileRankReversionStrategy:
    """
    Percentile Rank Reversion Strategy.
    
    Percentile rank of current close vs last 252 bars.
    BUY when percentile < 10 (extreme low).
    SELL when percentile > 90 (extreme high).
    Exit at 50th percentile.
    Confidence: |percentile - 50| / 50.
    """
    def __init__(self, data):
        self.df = data
        self.lookback = 252
        self.in_position = False
        self._calculate_percentile_rank()

    def _calculate_percentile_rank(self):
        # Percentile rank of current close vs last 252 bars
        # We use rolling apply for percentile rank
        def get_rank(window):
            if len(window) < 2: return np.nan
            # rank / (n-1) gives a value between 0 and 1
            return (window.argsort().argsort()[-1]) / (len(window) - 1) * 100

        self.df['percentile_rank'] = self.df['close'].rolling(window=self.lookback).apply(
            lambda x: (np.sum(x < x[-1]) / (len(x) - 1)) * 100 if len(x) > 1 else np.nan,
            raw=True
        )

    def next(self, bar):
        if bar < self.lookback or bar >= len(self.df):
            return {"type": "HOLD", "price": self.df['close'].iloc[bar] if bar < len(self.df) else 0.0, "confidence": 0.0, "metadata": {}}

        percentile = self.df['percentile_rank'].iloc[bar]
        current_close = self.df['close'].iloc[bar]
        
        if np.isnan(percentile):
             return {"type": "HOLD", "price": float(current_close), "confidence": 0.0, "metadata": {}}

        signal_type = "HOLD"
        confidence = abs(percentile - 50) / 50.0
        metadata = {"percentile": float(percentile)}

        if not self.in_position:
            if percentile < 10:
                signal_type = "BUY"
                self.in_position = True
        else:
            if percentile > 90:
                signal_type = "SELL"
                self.in_position = False
            elif 45 <= percentile <= 55: # Exit at 50th percentile
                signal_type = "SELL"
                self.in_position = False

        return {
            "type": signal_type,
            "price": float(current_close),
            "confidence": float(confidence),
            "metadata": metadata
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"lookback": self.lookback}
