import numpy as np

class ParabolicSARTrail:
    """
    Parabolic SAR strategy.
    SAR with acceleration factor 0.02, max 0.2.
    BUY when SAR flips below price (bullish).
    SELL when SAR flips above price (bearish).
    Confidence: distance between SAR and price / ATR.
    """
    def __init__(self, data):
        """
        Initialize with data.
        :param data: dict containing 'close', 'high', 'low' as numpy arrays.
        """
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.n = len(self.close)
        
        self.sar, self.trend = self._calculate_psar(self.high, self.low, 0.02, 0.2)
        self.atr = self._calculate_atr(self.high, self.low, self.close, 14)
        self.signals = []
        self.current_idx = 0

    def _calculate_psar(self, high, low, af_start, af_max):
        sar = np.zeros_like(high)
        trend = np.zeros_like(high) # 1 for bullish, -1 for bearish
        
        # Initial values
        sar[0] = low[0]
        trend[0] = 1
        
        for i in range(1, len(high)):
            af = af_start
            # We need to track the af for the current trend
            # This is a simplified single-pass version for the class structure
            # In a full implementation, af would be stateful per trend
            # For this task, we'll assume a simplified recurrence.
            pass

        # Since calculating PSAR in a single vectorized/loop way is complex for a single class 
        # and needs to be "self-contained, numpy only", I'll implement the standard loop.
        
        sar = np.zeros_like(high)
        trend = np.zeros_like(high) # 1 bullish, -1 bearish
        af = af_start
        
        # Start with a neutral/bullish assumption
        trend[0] = 1
        sar[0] = low[0]
        
        for i in range(1, len(high)):
            if trend[i-1] == 1:
                # Bullish trend
                sar[i] = sar[i-1] + af * (high[i-1] - sar[i-1])
                # Ensure SAR is below the low of previous bars
                sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                
                if low[i] < sar[i]:
                    # Flip to bearish
                    trend[i] = -1
                    sar[i] = high[i-1] # Standard flip behavior
                    af = af_start
                else:
                    trend[i] = 1
                    if high[i] > high[i-1]:
                        af = min(af + af_start, af_max)
                    else:
                        af = af # keep current af (simplified)
            else:
                # Bearish trend
                sar[i] = sar[i-1] + af * (high[i-1] - sar[i-1])
                sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                
                if high[i] > sar[i]:
                    # Flip to bullish
                    trend[i] = 1
                    sar[i] = low[i-1]
                    af = af_start
                else:
                    trend[i] = -1
                    if low[i] < low[i-1]:
                        af = min(af + af_start, af_max)
                    else:
                        af = af
        return sar, trend

    def _calculate_atr(self, high, low, close, period):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(close)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period + 1, len(close)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    def next(self, bar_idx):
        if bar_idx < 1 or bar_idx >= self.n:
            return None
            
        self.current_idx = bar_idx
        price = self.close[bar_idx]
        sar_val = self.sar[bar_idx]
        trend_val = self.trend[bar_idx]
        atr_val = self.atr[bar_idx] if self.atr[bar_idx] > 0 else 1e-9
        
        signal_type = "HOLD"
        confidence = 0.0
        
        # Flip detection
        if trend_val == 1 and self.trend[bar_idx-1] == -1:
            signal_type = "BUY"
            confidence = min(abs(price - sar_val) / atr_val, 1.0)
        elif trend_val == -1 and self.trend[bar_idx-1] == 1:
            signal_type = "SELL"
            confidence = min(abs(sar_val - price) / atr_val, 1.0)
            
        signal = {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {
                "sar": float(sar_val),
                "trend": int(trend_val)
            }
        }
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"af_start": 0.02, "af_max": 0.2}
