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

class IVRankReversal(BaseStrategy):
    """
    Strategy 10: IV Rank Reversal (using realized volatility as IV proxy)
    - Calculate rolling realized vol over iv_lookback=30
    - IV rank = (current_vol - min_vol) / (max_vol - min_vol) over lookback
    - IV rank > extreme_threshold (80) -> expect vol decrease, sell vol (mean-revert)
    - IV rank < (100 - threshold) -> expect vol increase, buy vol
    - Confidence: how extreme the IV rank is
    """
    def __init__(self, iv_lookback=30, extreme_threshold=80):
        self.iv_lookback = iv_lookback
        self.extreme_threshold = extreme_threshold
        self.signals = []
        self.data = None
        self.current_index = 0
        
        # State
        self.position = None # 'LONG', 'SHORT', None

    def init(self, data):
        self.data = data.copy()
        self.current_index = 0
        
        # Realized Volatility: Rolling std of log returns
        self.data['log_ret'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data['realized_vol'] = self.data['log_ret'].rolling(window=self.iv_lookback).std() * np.sqrt(252) # Annualized
        
        # IV Rank calculation (using a rolling window of the realized vol itself)
        # We need the min and max of the realized vol over the lookback
        self.data['vol_min'] = self.data['realized_vol'].rolling(window=self.iv_lookback).min()
        self.data['vol_max'] = self.data['realized_vol'].rolling(window=self.iv_lookback).max()
        
        self.data['iv_rank'] = (self.data['realized_vol'] - self.data['vol_min']) / (self.data['vol_max'] - self.data['vol_min']) * 100

    def next(self, bar) -> Signal:
        idx = self.current_index
        if idx >= len(self.data):
            return Signal('HOLD', 0.0, None, 0.0, {})

        price = bar['close']
        timestamp = bar['timestamp']
        iv_rank = bar['iv_rank']

        if np.isnan(iv_rank):
            self.current_index += 1
            return Signal('HOLD', price, timestamp, 0.0, {})

        signal_type = 'HOLD'
        confidence = 0.0
        metadata = {}

        # Entry Logic
        if self.position is None:
            # High IV Rank -> Expect Vol decrease -> "Sell Vol" (Short momentum/mean-revert)
            if iv_rank > self.extreme_threshold:
                signal_type = 'SELL'
                self.position = 'SHORT'
                confidence = iv_rank / 100.0
                metadata = {'reason': 'high_iv_rank', 'iv_rank': iv_rank}
            
            # Low IV Rank -> Expect Vol increase -> "Buy Vol" (Long breakout)
            elif iv_rank < (100 - self.extreme_threshold):
                signal_type = 'BUY'
                self.position = 'LONG'
                confidence = (100 - iv_rank) / 100.0
                metadata = {'reason': 'low_iv_rank', 'iv_rank': iv_rank}

        # Exit Logic (Mean reversion to median)
        elif self.position == 'SHORT' and iv_rank < 50:
            signal_type = 'HOLD' # Close short
            self.position = None
            metadata = {'reason': 'iv_normalized'}
        elif self.position == 'LONG' and iv_rank > 50:
            signal_type = 'HOLD' # Close long
            self.position = None
            metadata = {'reason': 'iv_normalized'}

        sig = Signal(signal_type, price, timestamp, confidence, metadata)
        if signal_type != 'HOLD':
            self.signals.append(sig)
        
        self.current_index += 1
        return sig

    def get_signals(self) -> List[Signal]:
        return self.signals

    def get_params(self) -> Dict:
        return {
            "iv_lookback": self.iv_lookback,
            "extreme_threshold": self.extreme_threshold
        }
