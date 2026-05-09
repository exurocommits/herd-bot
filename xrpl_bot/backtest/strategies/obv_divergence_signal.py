from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from .base_strategy import BaseStrategy, Signal

class OBVDivergenceSignal(BaseStrategy):
    """
    Strategy 11: OBV Divergence Signal
    Detects hidden bullish/bearish divergence between price and On-Balance Volume (OBV).
    """
    def __init__(self, obv_period: int = 14):
        super().__init__()
        self.obv_period = obv_period
        self.obv = None
        self.price_slope_window = 5
        self.obv_slope_window = obv_period

    def init(self, data: Dict[str, np.ndarray]):
        """
        data: dict with keys 'close', 'volume', 'timestamp'
        """
        self.data = data
        self.close = data['close']
        self.volume = data['volume']
        self.timestamp = data['timestamp']
        self.signals = []
        
        # Calculate OBV
        price_diff = np.diff(self.close, prepend=self.close[0])
        obv_changes = np.where(price_diff > 0, self.volume, 
                               np.where(price_diff < 0, -self.volume, 0))
        self.obv = np.cumsum(obv_changes)

    def next(self, bar_idx: int) -> Signal:
        if bar_idx < max(self.obv_period, self.price_slope_window):
            return Signal('HOLD', self.close[bar_idx], self.timestamp[bar_idx], 0.0)

        # Calculate slopes
        # OBV slope: (OBV[-1] - OBV[-period]) / period
        obv_val_now = self.obv[bar_idx]
        obv_val_past = self.obv[bar_idx - self.obv_period]
        obv_slope = (obv_val_now - obv_val_past) / self.obv_period

        # Price slope over small window
        price_now = self.close[bar_idx]
        price_past = self.close[bar_idx - self.price_slope_window]
        price_slope = (price_now - price_past) / self.price_slope_window

        signal_type = 'HOLD'
        confidence = 0.0

        # Hidden Bullish Divergence: Price flat/declining + OBV rising sharply
        if price_slope <= 0.001 and obv_slope > 0:
            # Check if OBV slope is "sharp" (arbitrary threshold for implementation)
            # In real use, this would be normalized. Here we use magnitude.
            signal_type = 'BUY'
            confidence = min(abs(obv_slope) / (abs(price_slope) + 1e-9), 1.0)

        # Hidden Bearish Divergence: Price flat/rising + OBV declining sharply
        elif price_slope >= -0.001 and obv_slope < 0:
            signal_type = 'SELL'
            confidence = min(abs(obv_slope) / (abs(price_slope) + 1e-9), 1.0)

        sig = Signal(
            type=signal_type,
            price=self.close[bar_idx],
            timestamp=self.timestamp[bar_idx],
            confidence=float(confidence),
            metadata={'obv_slope': float(obv_slope), 'price_slope': float(price_slope)}
        )
        self.signals.append(sig)
        return sig

    def get_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.type != 'HOLD']

    def get_params(self) -> dict:
        return {'obv_period': self.obv_period}
