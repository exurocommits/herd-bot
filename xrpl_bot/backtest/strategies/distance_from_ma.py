import numpy as np
import pandas as pd

class DistanceFromMAStrategy:
    """
    Distance from 50 SMA Strategy.
    
    Distance from 50 SMA measured in standard deviations.
    std = std(close - SMA, 50).
    BUY when distance < -2 std (oversold).
    SELL when > +2 std (overbought).
    Exit at 0 std (back at MA).
    Confidence: |distance| / (3 * std).
    """
    def __init__(self, data):
        self.df = data
        self.sma_period = 50
        self.in_position = False
        self._calculate_indicators()

    def _calculate_indicators(self):
        sma = self.df['close'].rolling(window=self.sma_period).mean()
        diff = self.df['close'] - sma
        # std = std(close- SMA, 50)
        std = diff.rolling(window=self.sma_period).std()
        
        self.df['sma'] = sma
        self.df['diff'] = diff
        self.df['std_dev'] = std
        self.df['z_score'] = diff / std

    def next(self, bar):
        if bar < self.sma_period or bar >= len(self.df):
            return {"type": "HOLD", "price": self.df['close'].iloc[bar] if bar < len(self.df) else 0.0, "confidence": 0.0, "metadata": {}}

        z_score = self.df['z_score'].iloc[bar]
        current_close = self.df['close'].iloc[bar]
        std = self.df['std_dev'].iloc[bar]

        if np.isnan(z_score) or std == 0:
            return {"type": "HOLD", "price": float(current_close), "confidence": 0.0, "metadata": {}}

        signal_type = "HOLD"
        confidence = abs(z_score) / (3.0 * std) if std > 0 else 0.0
        metadata = {"z_score": float(z_score), "sma": float(self.df['sma'].iloc[bar])}

        if not self.in_position:
            if z_score < -2:
                signal_type = "BUY"
                self.in_position = True
        else:
            if z_score > 2:
                signal_type = "SELL"
                self.in_position = False
            elif abs(z_score) < 0.1: # Exit at 0 std (back at MA)
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
        return {"sma_period": self.sma_period}
