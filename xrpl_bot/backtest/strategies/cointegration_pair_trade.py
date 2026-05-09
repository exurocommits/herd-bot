from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
from .base_strategy import BaseStrategy, Signal

class CointegrationPairTrade(BaseStrategy):
    """
    Strategy 13: Cointegration Pair Trade
    Trades the spread between two cointegrated assets.
    """
    def __init__(self, lookback: int = 100, z_threshold: float = 2.0):
        super().__init__()
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.hedge_ratio = 1.0
        self.spread_history = []

    def init(self, data: Dict[str, np.ndarray]):
        """
        data: dict with keys 'price_a', 'price_b', 'timestamp'
        """
        self.data = data
        self.price_a = data['price_a']
        self.price_b = data['price_b']
        self.timestamp = data['timestamp']
        self.signals = []
        self.spreads = []

    def _calculate_spread(self, end_idx: int):
        if end_idx < self.lookback:
            return None

        start_idx = end_idx - self.lookback
        y = self.price_a[start_idx : end_idx + 1]
        x = self.price_b[start_idx : end_idx + 1]

        # Rolling OLS for hedge ratio
        # y = alpha + beta * x
        # Using numpy polyfit for simple linear regression
        coeffs = np.polyfit(x, y, 1)
        self.hedge_ratio = coeffs[0]
        alpha = coeffs[1]

        spread = self.price_a[end_idx] - (self.hedge_ratio * self.price_b[end_idx] + alpha)
        return spread

    def next(self, bar_idx: int) -> Signal:
        spread = self._calculate_spread(bar_idx)
        if spread is None:
            return Signal('HOLD', self.price_a[bar_idx], self.timestamp[bar_idx], 0.0)

        # We need a history of spreads to calculate Z-score
        # Note: For performance in real systems, we'd maintain a rolling window
        # Here we re-calculate/approximate for clarity
        
        # In a real implementation, we'd store the calculated spread for each index
        # Since we can't easily 'look back' at previous 'next' calls without storage:
        if not hasattr(self, '_calculated_spreads'):
            self._calculated_spreads = np.array([])

        # This is a simplified way to build the spread series for Z-score calculation
        # In actual backtest, spread is calculated at each step.
        # Let's simulate the spread history for the window.
        
        # For the sake of the 'next' pattern, we'll compute the Z-score manually
        # by looking at recent spreads.
        
        # [Optimization: This part is slightly heavy for every 'next' call]
        # Let's assume we maintain a rolling buffer of spreads.
        if not hasattr(self, '_spread_buffer'):
            self._spread_buffer = []
        
        self._spread_buffer.append(spread)
        if len(self._spread_buffer) > self.lookback:
            self._spread_buffer.pop(0)
            
        if len(self._spread_buffer) < self.lookback:
            return Signal('HOLD', self.price_a[bar_idx], self.timestamp[bar_idx], 0.0)

        spread_arr = np.array(self._spread_buffer)
        mean_spread = np.mean(spread_arr)
        std_spread = np.std(spread_arr)
        
        if std_spread == 0:
            return Signal('HOLD', self.price_a[bar_idx], self.timestamp[bar_idx], 0.0)
            
        z_score = (spread - mean_spread) / std_spread
        
        signal_type = 'HOLD'
        confidence = 0.0

        # Z-score > threshold -> SELL spread (short A, long B)
        if z_score > self.z_threshold:
            signal_type = 'SELL'
            confidence = min(abs(z_score) / self.z_threshold, 1.0)
        # Z-score < -threshold -> BUY spread (long A, short B)
        elif z_score < -self.z_threshold:
            signal_type = 'BUY'
            confidence = min(abs(z_score) / self.z_threshold, 1.0)
            
        # Z-score crosses 0 -> close (In a real bot, this would trigger a CLOSE signal)
        # For this implementation, we'll treat it as a signal that can revert
        
        sig = Signal(
            type=signal_type,
            price=self.price_a[bar_idx],
            timestamp=self.timestamp[bar_idx],
            confidence=float(confidence),
            metadata={'z_score': float(z_score), 'hedge_ratio': float(self.hedge_ratio)}
        )
        self.signals.append(sig)
        return sig

    def get_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.type != 'HOLD']

    def get_params(self) -> dict:
        return {'lookback': self.lookback, 'z_threshold': self.z_threshold}
