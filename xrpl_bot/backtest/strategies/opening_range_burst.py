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

class OpeningRangeBurst(BaseStrategy):
    \"\"\"
    Strategy 7: Opening Range Burst
    - Define opening range from first range_minutes=30 bars of each day
    - Price breaks above range high + volume > 1.5x avg -> BUY
    - Stop at range low, target at 2x risk (risk_reward=2.0)
    - Confidence: range size relative to ATR + volume surge magnitude
    \"\"\"
    def __init__(self, range_minutes=30, volume_multiplier=1.5, risk_reward=2.0, atr_period=14):
        self.range_minutes = range_minutes
        self.volume_multiplier = volume_multiplier
        self.risk_reward = risk_reward
        self.atr_period = atr_period
        self.signals = []
        self.data = None
        self.current_index = 0
        
        # State
        self.position = None # 'LONG', None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Daily state
        self.current_day = None
        self.range_high = 0.0
        self.range_low = 0.0
        self.range_end_timestamp = None
        self.avg_volume = 0.0

    def init(self, data):
        self.data = data.copy()
        self.current_index = 0
        
        # Pre-calculate ATR
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=self.atr_period).mean()
        
        # Avg volume (rolling 20)
        self.data['avg_volume'] = self.data['volume'].rolling(window=20).mean()
        
        # Identify days and range end timestamps
        self.data['date'] = self.data['timestamp'].dt.date
        
        # Pre-calculate the end of the opening range for each day
        day_groups = self.data.groupby('date')
        range_ends = {}
        for date, group in day_groups:
            first_ts = group['timestamp'].iloc[0]
            range_ends[date] = first_ts + pd.Timedelta(minutes=self.range_minutes)
        self.range_ends = range_ends

    def next(self, bar) -> Signal:
        idx = self.current_index
        if idx >= len(self.data):
            return Signal('HOLD', 0.0, None, 0.0, {})

        price = bar['close']
        timestamp = bar['timestamp']
        date = bar['date']
        volume = bar['volume']
        atr = bar['atr']
        avg_vol = bar['avg_volume']

        # 1. Handle Day Change / Reset State
        if date != self.current_day:
            self.current_day = date
            self.range_high = 0.0
            self.range_low = float('inf')
            self.range_end_timestamp = self.range_ends.get(date)
            self.position = None
            self.entry_price = 0.0

        # 2. Build Opening Range
        if timestamp <= self.range_end_timestamp:
            self.range_high = max(self.range_high, bar['high'])
            self.range_low = min(self.range_low, bar['low'])
            self.current_index += 1
            return Signal('HOLD', price, timestamp, 0.0, {'phase': 'range_building'})

        # 3. Trading Phase (After Opening Range)
        signal_type = 'HOLD'
        confidence = 0.0
        metadata = {'phase': 'trading'}

        # Exit Logic (Stop Loss or Take Profit)
        if self.position == 'LONG':
            if price <= self.stop_loss:
                signal_type = 'SELL'
                self.position = None
                metadata['reason'] = 'stop_loss'
            elif price >= self.take_profit:
                signal_type = 'SELL'
                self.position = None
                metadata['reason'] = 'take_profit'

        # Entry Logic
        if self.position is None:
            # Price breaks above range high + volume > 1.5x avg
            if price > self.range_high and volume > (self.volume_multiplier * avg_vol):
                self.position = 'LONG'
                self.entry_price = price
                risk = self.entry_price - self.range_low
                if risk > 0:
                    self.stop_loss = self.range_low
                    self.take_profit = self.entry_price + (self.risk_reward * risk)
                    
                    signal_type = 'BUY'
                    # Confidence: range size relative to ATR + volume surge magnitude
                    range_size_pct = (self.range_high - self.range_low) / self.entry_price if self.entry_price > 0 else 0
                    atr_pct = atr / self.entry_price if self.entry_price > 0 else 1
                    vol_surge = volume / avg_vol if avg_vol > 0 else 1
                    confidence = min((range_size_pct / atr_pct if atr_pct > 0 else 0) * (vol_surge / self.volume_multiplier), 1.0)
                    metadata = {'reason': 'range_breakout', 'range_high': self.range_high, 'range_low': self.range_low}
                else:
                    # Avoid invalid stop/profit
                    self.position = None

        sig = Signal(signal_type, price, timestamp, confidence, metadata)
        if signal_type != 'HOLD':
            self.signals.append(sig)
        
        self.current_index += 1
        return sig

    def get_signals(self) -> List[Signal]:
        return self.signals

    def get_params(self) -> Dict:
        return {
            "range_minutes": self.range_minutes,
            "volume_multiplier": self.volume_multiplier,
            "risk_reward": self.risk_reward,
            "atr_period": self.atr_period
        }
