import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

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

class VWAPMeanReversion(BaseStrategy):
    """
    Strategy 6: VWAP Mean Reversion
    - Price below VWAP * (1 - deviation_threshold) -> BUY
    - Price above VWAP * (1 + deviation_threshold) -> SELL
    - Price crosses VWAP -> close (HOLD)
    - Confidence: distance from VWAP as % of ATR
    """
    def __init__(self, deviation_threshold=0.02, atr_period=14):
        self.deviation_threshold = deviation_threshold
        self.atr_period = atr_period
        self.signals = []
        self.data = None
        self.current_index = 0
        
        # State
        self.position = None  # 'LONG', 'SHORT', None
        self.entry_price = 0.0

    def init(self, data):
        """
        data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        """
        self.data = data
        self.current_index = 0
        
        # Pre-calculate VWAP
        self.data['vwap'] = (self.data['volume'] * (self.data['high'] + self.data['low'] + self.data['close']) / 3).cumsum() / self.data['volume'].cumsum()
        
        # Pre-calculate ATR
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=self.atr_period).mean()

    def next(self, bar) -> Signal:
        idx = self.current_index
        if idx >= len(self.data):
            return Signal('HOLD', 0.0, None, 0.0, {})

        price = bar['close']
        vwap = bar['vwap']
        atr = bar['atr']
        timestamp = bar['timestamp']

        if np.isnan(vwap) or np.isnan(atr):
            self.current_index += 1
            return Signal('HOLD', price, timestamp, 0.0, {})

        signal_type = 'HOLD'
        confidence = 0.0
        metadata = {}

        # Check for closing position (crosses VWAP)
        if self.position == 'LONG' and price > vwap:
            signal_type = 'HOLD'
            self.position = None
        elif self.position == 'SHORT' and price < vwap:
            signal_type = 'HOLD'
            self.position = None

        # Entry logic
        if self.position is None:
            # Buy if price < VWAP * (1 - threshold)
            if price < vwap * (1 - self.deviation_threshold):
                signal_type = 'BUY'
                self.position = 'LONG'
                self.entry_price = price
                # Confidence: distance from VWAP as % of ATR
                dist = (vwap - price) / vwap
                confidence = min(dist / (atr / price) if atr > 0 else 0, 1.0)
                metadata = {'reason': 'below_vwap_threshold'}
            
            # Sell if price > VWAP * (1 + threshold)
            elif price > vwap * (1 + self.deviation_threshold):
                signal_type = 'SELL'
                self.position = 'SHORT'
                self.entry_price = price
                dist = (price - vwap) / vwap
                confidence = min(dist / (atr / price) if atr > 0 else 0, 1.0)
                metadata = {'reason': 'above_vwap_threshold'}
        
        sig = Signal(signal_type, price, timestamp, confidence, metadata)
        if signal_type != 'HOLD':
            self.signals.append(sig)
        
        self.current_index += 1
        return sig

    def get_signals(self) -> List[Signal]:
        return self.signals

    def get_params(self) -> Dict:
        return {
            "deviation_threshold": self.deviation_threshold,
            "atr_period": self.atr_period
        }

# Note: imports like pandas would be needed in the real implementation
import pandas as pd
