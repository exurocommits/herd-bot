from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from .base_strategy import BaseStrategy, Signal

class VolumeProfileNode(BaseStrategy):
    """
    Strategy 12: Volume Profile Node
    Identifies High-Volume Nodes (HVN) and trades bounces/resistance.
    """
    def __init__(self, profile_rows: int = 50, lookback_days: int = 2):
        super().__init__()
        self.profile_rows = profile_rows
        self.lookback_days = lookback_days
        self.hvn_price = None
        self.hvn_volume = 0.0

    def init(self, data: Dict[str, np.ndarray]):
        """
        data: dict with keys 'close', 'volume', 'timestamp'
        Note: lookback_days is assumed to be handled by the data slice passed in, 
        or we calculate index-based lookback if data is 1-day resolution.
        """
        self.data = data
        self.close = data['close']
        self.volume = data['volume']
        self.timestamp = data['timestamp']
        self.signals = []

    def _calculate_hvn(self, end_idx: int):
        # Approximate lookback (assuming 1 bar = 1 day for simplicity in this implementation, 
        # or user provides appropriate window). 
        # A better way is to use timestamps to find the window.
        lookback = self.lookback_days 
        if end_idx < lookback:
            return
        
        start_idx = max(0, end_idx - lookback)
        prices = self.close[start_idx:end_idx+1]
        volumes = self.volume[start_idx:end_idx+1]
        
        if len(prices) == 0:
            return

        min_p, max_p = np.min(prices), np.max(prices)
        if min_p == max_p:
            return
            
        bins = np.linspace(min_p, max_p, self.profile_rows)
        # Calculate volume per bin
        bin_volumes = np.zeros(self.profile_rows)
        for i in range(len(prices)):
            # Find which bin the price falls into
            if prices[i] <= bins[0]:
                idx = 0
            elif prices[i] >= bins[-1]:
                idx = self.profile_rows - 1
            else:
                idx = np.searchsorted(bins, prices[i]) - 1
            bin_volumes[idx] += volumes[i]

        avg_vol = np.mean(bin_volumes)
        hvn_idx = np.argmax(bin_volumes)
        
        if bin_volumes[hvn_idx] > 2 * avg_vol:
            self.hvn_price = (bins[hvn_idx] + bins[hvn_idx+1]) / 2 if hvn_idx < self.profile_rows-1 else bins[hvn_idx]
            self.hvn_volume = bin_volumes[hvn_idx]
            self.avg_bin_vol = avg_vol
        else:
            self.hvn_price = None

    def next(self, bar_idx: int) -> Signal:
        self._calculate_hvn(bar_idx)
        
        if self.hvn_price is None:
            return Signal('HOLD', self.close[bar_idx], self.timestamp[bar_idx], 0.0)

        price = self.close[bar_idx]
        prev_price = self.close[bar_idx-1] if bar_idx > 0 else price
        
        signal_type = 'HOLD'
        confidence = 0.0

        # Price approaching HVN from above -> BUY (support)
        if prev_price > self.hvn_price >= price:
            signal_type = 'BUY'
            confidence = min(self.hvn_volume / (self.avg_bin_vol * 5), 1.0) # Example normalization

        # Price approaching HVN from below -> SELL (resistance)
        elif prev_price < self.hvn_price <= price:
            signal_type = 'SELL'
            confidence = min(self.hvn_volume / (self.avg_bin_vol * 5), 1.0)

        sig = Signal(
            type=signal_type,
            price=price,
            timestamp=self.timestamp[bar_idx],
            confidence=float(confidence),
            metadata={'hvn_price': float(self.hvn_price), 'hvn_vol': float(self.hvn_volume)}
        )
        self.signals.append(sig)
        return sig

    def get_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.type != 'HOLD']

    def get_params(self) -> dict:
        return {'profile_rows': self.profile_rows, 'lookback_days': self.lookback_days}
