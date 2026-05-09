import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Signal:
    type: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    timestamp: Any
    confidence: float
    metadata: Dict[str, Any]

class BaseStrategy:
    def init(self, data):
        raise NotImplementedError

    def next(self, bar) -> Signal:
        raise NotImplementedError

    def get_signals(self) -> List[Signal]:
        raise NotImplementedError

    def get_params(self) -> Dict:
        raise NotImplementedError

class VolumeSurgeBreakout(BaseStrategy):
    """
    Strategy 8: Volume Surge Breakout
    - Track volume MA over volume_ma_period=20
    - Price breaks 20-bar high + volume > volume_multiplier (2.0x) avg -> BUY
    - Volume fades below 0.5x avg -> SELL/close
    - Confidence: volume surge ratio * breakout percentage
    """
    def __init__(self, volume_ma_period=20, volume_multiplier=2.0, volume_fade_threshold=0.5, breakout_period=20):
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.volume_fade_threshold = volume_fade_threshold
        self.breakout_period = breakout_period
        self.signals = []
        self.data = None
        self.current_index = 0
        
        # State
        self.position = None # 'LONG', None
        self.entry_price = 0.0

    def init(self, data):
        self.data = data.copy()
        self.current_index = 0
        
        # Volume MA
        self.data['vol_ma'] = self.data['volume'].rolling(window=self.volume_ma_period).mean()
        
        # Price High MA (for breakout)
        self.data['price_high_ma'] = self.data['high'].shift(1).rolling(window=self.breakout_period).max()

    def next(self, bar) -> Signal:
        idx = self.current_index
        if idx >= len(self.data):
            return Signal('HOLD', 0.0, None, 0.0, {})

        price = bar['close']
        timestamp = bar['timestamp']
        volume = bar['volume']
        vol_ma = bar['vol_ma']
        price_high = bar['price_high_ma']

        if np.isnan(vol_ma) or np.isnan(price_high):
            self.current_index += 1
            return Signal('HOLD', price, timestamp, 0.0, {})

        signal_type = 'HOLD'
        confidence = 0.0
        metadata = {}

        # 1. Check Exit (Volume Fades)
        if self.position == 'LONG':
            if volume < (self.volume_fade_threshold * vol_ma):
                signal_type = 'SELL'
                self.position = None
                metadata = {'reason': 'volume_fade'}

        # 2. Check Entry (Breakout + Volume Surge)
        elif self.position is None:
            if price > price_high and volume > (self.volume_multiplier * vol_ma):
                self.position = 'LONG'
                self.entry_price = price
                signal_type = 'BUY'
                
                # Confidence: volume surge ratio * breakout percentage
                vol_surge_ratio = volume / vol_ma if vol_ma > 0 else 1.0
                breakout_pct = (price - price_high) / price_high if price_high > 0 else 0.0
                confidence = min(vol_surge_ratio * (1.0 + breakout_pct), 1.0)
                metadata = {'reason': 'breakout', 'vol_surge': vol_surge_ratio, 'breakout_pct': breakout_pct}

        sig = Signal(signal_type, price, timestamp, confidence, metadata)
        if signal_type != 'HOLD':
            self.signals.append(sig)
        
        self.current_index += 1
        return sig

    def get_signals(self) -> List[Signal]:
        return self.signals

    def get_params(self) -> Dict:
        return {
            "volume_ma_period": self.volume_ma_period,
            "volume_multiplier": self.volume_multiplier,
            "volume_fade_threshold": self.volume_fade_threshold,
            "breakout_period": self.breakout_period
        }
