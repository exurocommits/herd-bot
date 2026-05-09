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

class ATRSqueezeExpansion(BaseStrategy):
    """
    Strategy 9: ATR Squeeze Expansion
    - Calculate ATR (period=14) and its own moving average
    - When ATR < squeeze_threshold (0.01 * close) -> squeeze detected
    - When squeeze releases (ATR crosses above its MA) + price breaks 20-bar high -> BUY
    - When ATR expansion peaks (ATR > 2x its MA) -> SELL/close
    - Confidence: squeeze duration * expansion strength
    """
    def __init__(self, atr_period=14, squeeze_threshold_factor=0.01, breakout_period=20):
        self.atr_period = atr_period
        self.squeeze_threshold_factor = squeeze_threshold_factor
        self.breakout_period = breakout_period
        self.signals = []
        self.data = None
        self.current_index = 0
        
        # State
        self.position = None # 'LONG', None
        self.squeeze_active = False
        self.squeeze_start_idx = 0
        self.entry_price = 0.0

    def init(self, data):
        self.data = data.copy()
        self.current_index = 0
        
        # ATR
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=self.atr_period).mean()
        
        # ATR Moving Average
        self.data['atr_ma'] = self.data['atr'].rolling(window=20).mean()
        
        # Price breakout high (20-bar)
        self.data['price_high_20'] = self.data['high'].shift(1).rolling(window=self.breakout_period).max()

    def next(self, bar) -> Signal:
        idx = self.current_index
        if idx >= len(self.data):
            return Signal('HOLD', 0.0, None, 0.0, {})

        price = bar['close']
        timestamp = bar['timestamp']
        atr = bar['atr']
        atr_ma = bar['atr_ma']
        price_high = bar['price_high_20']

        if np.isnan(atr) or np.isnan(atr_ma) or np.isnan(price_high):
            self.current_index += 1
            return Signal('HOLD', price, timestamp, 0.0, {})

        signal_type = 'HOLD'
        confidence = 0.0
        metadata = {}

        # 1. Detection Logic
        is_squeezed = atr < (self.squeeze_threshold_factor * price)
        
        # Squeeze Detection
        if is_squeezed and not self.squeeze_active:
            self.squeeze_active = True
            self.squeeze_start_idx = idx
        elif not is_squeezed and self.squeeze_active:
            # Squeeze just released
            # Check if price breaks 20-bar high
            if price > price_high:
                self.position = 'LONG'
                self.entry_price = price
                signal_type = 'BUY'
                
                # Confidence: squeeze duration * expansion strength
                squeeze_duration = idx - self.squeeze_start_idx
                expansion_strength = atr / atr_ma if atr_ma > 0 else 1.0
                confidence = min(squeeze_duration * (expansion_strength / 2.0), 1.0)
                metadata = {'reason': 'squeeze_release_breakout', 'squeeze_duration': squeeze_duration}
                self.signals.append(Signal(signal_type, price, timestamp, confidence, metadata))
            
            self.squeeze_active = False

        # 2. Exit Logic (Expansion Peaks)
        if self.position == 'LONG':
            if atr > (2.0 * atr_ma):
                signal_type = 'SELL'
                self.position = None
                metadata = {'reason': 'atr_expansion_peak'}

        sig = Signal(signal_type, price, timestamp, confidence, metadata)
        if signal_type != 'HOLD':
            self.signals.append(sig)
        
        self.current_index += 1
        return sig

    def get_signals(self) -> List[Signal]:
        return self.signals

    def get_params(self) -> Dict:
        return {
            "atr_period": self.atr_period,
            "squeeze_threshold_factor": self.squeeze_threshold_factor,
            "breakout_period": self.breakout_period
        }
