import numpy as np
import pandas as pd

class BollingerSqueezeStrategy:
    """
    Bollinger Squeeze Strategy.
    
    BB width = (upper - lower) / middle.
    Track 120-bar min width.
    When width at 120-bar low = squeeze.
    BUY on close above upper BB after squeeze.
    SELL on BB width expansion peak (width > 2x recent min).
    Confidence: squeeze duration * breakout strength.
    """
    def __init__(self, data):
        self.df = data
        self.period = 20
        self.std_dev = 2
        self.lookback = 120
        self.is_squeezed = False
        self.squeeze_start_bar = None
        
        self._calculate_indicators()

    def _calculate_indicators(self):
        close = self.df['close']
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        
        self.df['bb_upper'] = sma + (self.std_dev * std)
        self.df['bb_lower'] = sma - (self.std_dev * std)
        self.df['bb_mid'] = sma
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_mid']
        
        # Min width over 120 bars
        self.df['bb_width_min'] = self.df['bb_width'].rolling(window=self.lookback).min()

    def next(self, bar):
        if bar < self.lookback or bar >= len(self.df):
            return {"type": "HOLD", "price": self.df['close'].iloc[bar] if bar < len(self.df) else 0.0, "confidence": 0.0, "metadata": {}}

        current_close = self.df['close'].iloc[bar]
        current_width = self.df['bb_width'].iloc[bar]
        min_width = self.df['bb_width_min'].iloc[bar]
        upper_bb = self.df['bb_upper'].iloc[bar]
        
        signal_type = "HOLD"
        confidence = 0.0
        metadata = {}

        # Squeeze detection
        is_currently_squeezed = np.isclose(current_width, min_width, atol=1e-6)
        
        if is_currently_squeezed:
            if not self.is_squeezed:
                self.is_squeezed = True
                self.squeeze_start_bar = bar
        else:
            if self.is_squeezed:
                # Squeeze just ended - check for breakout
                if current_close > upper_bb:
                    squeeze_duration = bar - self.squeeze_start_bar
                    breakout_strength = (current_close - self.df['bb_mid'].iloc[bar]) / self.df['bb_mid'].iloc[bar]
                    signal_type = "BUY"
                    confidence = float(squeeze_duration * max(breakout_strength, 0.001))
                    metadata = {"squeeze_duration": squeeze_duration, "breakout_strength": float(breakout_strength)}
                
                self.is_squeezed = False
                self.squeeze_start_bar = None

        # Sell condition: width expansion peak (width > 2x recent min)
        # We track the most recent min width for the sell condition
        if signal_type == "HOLD":
            recent_min = self.df['bb_width'].iloc[max(0, bar-20):bar].min()
            if current_width > 2 * recent_min and recent_min > 0:
                # This is a simplified "expansion peak" detection
                signal_type = "SELL"
                confidence = 0.5 # Default confidence for sell
                metadata = {"expansion_ratio": float(current_width / recent_min)}

        return {
            "type": signal_type,
            "price": float(current_close),
            "confidence": float(confidence),
            "metadata": metadata
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"period": self.period, "std_dev": self.std_dev, "lookback": self.lookback}
