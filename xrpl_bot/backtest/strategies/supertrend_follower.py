import numpy as np
from typing import Dict, Any, List
from .base_strategy import BaseStrategy, Signal

class SupertrendFollower(BaseStrategy):
    """
    Strategy 2: Supertrend Follower
    Calculate Supertrend indicator (period=10, multiplier=3.0)
    Price closes above Supertrend -> BUY
    Price closes below Supertrend -> SELL
    Supertrend = (high+low)/2 ± multiplier*ATR
    Confidence: distance from Supertrend line / ATR
    """
    def __init__(self, period=10, multiplier=3.0):
        super().__init__()
        self.period = period
        self.multiplier = multiplier
        self.supertrend = None
        self.direction = None  # 1 for up, -1 for down
        self.atr = None

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(tr)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period + 1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    def init(self, data: Dict[str, np.ndarray]):
        super().init(data)
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        self.atr = self._calculate_atr(high, low, close, self.period)
        hl2 = (high + low) / 2
        
        self.supertrend = np.zeros_like(high)
        self.direction = np.zeros(len(high), dtype=int)
        
        curr_direction = 1
        for i in range(1, len(high)):
            upper_band = hl2[i] + (self.multiplier * self.atr[i])
            lower_band = hl2[i] - (self.multiplier * self.atr[i])
            
            if i < self.period:
                continue
                
            if close[i] > self.supertrend[i-1] and close[i-1] <= self.supertrend[i-1] and self.supertrend[i-1] != 0:
                # This is a simplification, usually Supertrend logic is more stateful
                pass 

            # Real Supertrend logic
            if close[i] > upper_band: # This is not how ST works, let's do it properly
                pass

        # Let's implement the standard ST logic more carefully
        st_upper = np.zeros_like(high)
        st_lower = np.zeros_like(high)
        st_val = np.zeros_like(high)
        st_dir = np.ones(len(high), dtype=int) # 1 up, -1 down

        hl2_arr = (high + low) / 2
        
        for i in range(1, len(high)):
            # Upper Band
            ub = hl2_arr[i] + self.multiplier * self.atr[i]
            if ub < st_upper[i-1] or close[i-1] > st_upper[i-1]:
                st_upper[i] = ub
            else:
                st_upper[i] = st_upper[i-1]
                
            # Lower Band
            lb = hl2_arr[i] - self.multiplier * self.atr[i]
            if lb > st_lower[i-1] or close[i-1] < st_lower[i-1]:
                st_lower[i] = lb
            else:
                st_lower[i] = st_lower[i-1]
            
            # Direction
            if close[i] > st_upper[i-1]:
                st_dir[i] = 1
            elif close[i] < st_lower[i-1]:
                st_dir[i] = -1
            else:
                st_dir[i] = st_dir[i-1]
                
            st_val[i] = st_lower[i] if st_dir[i] == 1 else st_upper[i]

        self.supertrend = st_val
        self.direction = st_dir

    def next(self, bar: Dict[str, Any]) -> Signal:
        idx = self.current_index
        if idx >= len(self.data['close']):
            return Signal('HOLD', bar['close'], bar['timestamp'], 0.0)

        current_price = bar['close']
        st_val = self.supertrend[idx]
        st_dir = self.direction[idx]
        atr_val = self.atr[idx]
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        if st_dir == 1 and current_price > st_val:
            # We are in uptrend, check if we just crossed
            # For simplicity, if st_dir is 1, we BUY (or hold BUY)
            # But the task says: "Price closes above Supertrend -> BUY"
            # Let's check if the direction just changed
            pass

        # Refined logic:
        # If direction is 1 (up), we are in a BUY state. 
        # If direction is -1 (down), we are in a SELL state.
        # To avoid continuous signals, we only signal on change or if it's the first one.
        # But the requirement is simple: "Price closes above Supertrend -> BUY"
        
        if st_dir == 1:
            signal_type = 'BUY'
            dist = abs(current_price - st_val)
            confidence = min(1.0, dist / (atr_val if atr_val > 0 else 1.0))
        elif st_dir == -1:
            signal_type = 'SELL'
            dist = abs(current_price - st_val)
            confidence = min(1.0, dist / (atr_val if atr_val > 0 else 1.0))

        sig = Signal(
            type=signal_type,
            price=current_price,
            timestamp=bar['timestamp'],
            confidence=confidence,
            metadata={'supertrend': st_val, 'direction': st_dir}
        )
        self.signals.append(sig)
        return sig

    def get_params(self) -> Dict[str, Any]:
        return {'period': self.period, 'multiplier': self.multiplier}
