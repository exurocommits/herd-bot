from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from .base_strategy import BaseStrategy, Signal

class ZScoreSpreadMeanRev(BaseStrategy):
    """
    Strategy 14: Z-Score Spread Mean Reversion
    Simpler version of pair trading using the ratio of two prices.
    """
    def __init__(self, window: int = 50, z_entry: float = -2.0, z_exit: float = 0.0):
        super().__init__()
        self.window = window
        self.z_entry = z_entry # e.g. -2.0
        self.z_exit = z_exit   # e.g. 0.0
        self.ratio_history = []

    def init(self, data: Dict[str, np.ndarray]):
        """
        data: dict with keys 'price_a', 'price_b', 'timestamp'
        """
        self.data = data
        self.price_a = data['price_a']
        self.price_b = data['price_b']
        self.timestamp = data['timestamp']
        self.signals = []
        self.ratio_history = []

    def next(self, bar_idx: int) -> Signal:
        if bar_idx == 0:
            return Signal('HOLD', self.price_a[bar_idx], self.timestamp[bar_idx], 0.0)

        # Calculate spread ratio
        ratio = self.price_a[bar_idx] / self.price_b[bar_idx]
        self.ratio_history.append(ratio)
        
        if len(self.ratio_history) < self.window:
            return Signal('HOLD', self.price_a[bar_idx], self.timestamp[bar_idx], 0.0)

        # Keep only window
        if len(self.ratio_history) > self.window:
            self.ratio_history.pop(0)

        # Calculate Z-score
        window_data = np.array(self.ratio_history)
        mean_ratio = np.mean(window_data)
        std_ratio = np.std(window_data)
        
        if std_ratio == 0:
            return Signal('HOLD', self.price_a[bar_idx], self.timestamp[bar_idx], 0.0)
            
        z_score = (ratio - mean_ratio) / std_ratio
        
        signal_type = 'HOLD'
        confidence = 0.0

        # Z < z_entry (-2.0) -> BUY the ratio (long A, short B)
        if z_score < self.z_entry:
            signal_type = 'BUY'
            confidence = min(abs(z_score) / abs(self.z_entry), 1.0)
        # Z > -z_entry (2.0) -> SELL the ratio (short A, long B)
        # Note: Task said Z > -z_entry. If z_entry is -2.0, then -z_entry is 2.0.
        elif z_score > -self.z_entry:
            signal_type = 'SELL'
            confidence = min(abs(z_score) / abs(self.z_entry), 1.0)
            
        # Z crosses z_exit (0) -> close 
        # (In practice, we'd handle closing logic, but for this class we'll return HOLD if it's in the exit zone)
        if abs(z_score) < 0.1: # threshold for crossing zero
             signal_type = 'HOLD'

        sig = Signal(
            type=signal_type,
            price=self.price_a[bar_idx],
            timestamp=self.timestamp[bar_idx],
            confidence=float(confidence),
            metadata={'z_score': float(z_score), 'ratio': float(ratio)}
        )
        self.signals.append(sig)
        return sig

    def get_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.type != 'HOLD']

    def get_params(self) -> dict:
        return {'window': self.window, 'z_entry': self.z_entry, 'z_exit': self.z_exit}
