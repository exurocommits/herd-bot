import numpy as np

class EhlersSuperSmoother:
    """
    2-pole Ehlers super smoother: 
    ssf = (a1*close + a2*prev_ssf1 - b2*prev_ssf2) 
    where a1,b1,b2 from cutoff period (default 10).
    BUY on close crossing above smoothed line.
    SELL on cross below.
    Confidence: |close - ssf| / ATR.
    """
    def __init__(self, data):
        self.close = np.array(data['close'], dtype=float)
        self.high = np.array(data['high'], dtype=float)
        self.low = np.array(data['low'], dtype=float)
        self.cutoff = 10
        self.signals = []
        self.current_idx = len(self.close) - 1
        
        # State for the 2-pole filter
        self.ssf_history = []
        self.price_history = list(self.close)
        
        # Coefficients for a 10-period cutoff (approximate)
        # Following the prompt's specific formula: ssf = (a1*close + a2*prev_ssf1 - b2*prev_ssf2)
        self.a1 = 0.067455
        self.a2 = 0.283592
        self.b2 = -0.530697

        # Initialize filter by running through historical data
        if len(self.close) > 0:
            # We'll bootstrap the filter with the first available price
            self.ssf_history = [self.close[0]]
            for i in range(1, len(self.close)):
                prev_ssf1 = self.ssf_history[-1]
                prev_ssf2 = self.ssf_history[-2] if len(self.ssf_history) > 1 else prev_ssf1
                val = (self.a1 * self.close[i]) + (self.a2 * prev_ssf1) - (self.b2 * prev_ssf2)
                self.ssf_history.append(val)

    def _calculate_atr(self, period=14):
        if len(self.high) < period + 1:
            return 0.0
        
        # Calculate TR (True Range)
        # TR = max(H-L, abs(H-Cp), abs(L-Cp))
        h = self.high
        l = self.low
        c = self.close
        
        # We only want the last 'period' bars
        # To avoid large array ops in 'next', we use the tail
        idx = len(h)
        tr_list = []
        for i in range(max(1, idx - period), idx):
            val = max(h[i] - l[i], 
                      abs(h[i] - c[i-1]), 
                      abs(l[i] - c[i-1]))
            tr_list.append(val)
            
        if not tr_list: return 0.0
        return float(np.mean(tr_list))

    def next(self, bar):
        # 1. Update data
        self.close = np.append(self.close, bar['close'])
        self.high = np.append(self.high, bar['high'])
        self.low = np.append(self.low, bar['low'])
        self.price_history.append(bar['close'])
        self.current_idx += 1

        # 2. Update SSF Filter
        prev_ssf1 = self.ssf_history[-1]
        prev_ssf2 = self.ssf_history[-2] if len(self.ssf_history) > 1 else prev_ssf1
        current_ssf = (self.a1 * bar['close']) + (self.a2 * prev_ssf1) - (self.b2 * prev_ssf2)
        self.ssf_history.append(current_ssf)

        # 3. Calculate Indicators
        price = float(bar['close'])
        atr = self._calculate_atr()
        conf = 0.0
        if atr > 0:
            conf = float(np.clip(abs(price - current_ssf) / atr, 0, 1))

        # 4. Signal Logic (Crossing)
        signal_type = "HOLD"
        
        # Need previous price and previous ssf to detect crossing
        if len(self.price_history) >= 2 and len(self.ssf_history) >= 2:
            prev_price = self.price_history[-2]
            prev_ssf = self.ssf_history[-2]
            
            # BUY: close crosses above smoothed line
            if prev_price <= prev_ssf and price > current_ssf:
                signal_type = "BUY"
            # SELL: close crosses below smoothed line
            elif prev_price >= prev_ssf and price < current_ssf:
                signal_type = "SELL"

        signal = {"type": signal_type, "price": price, "confidence": conf, "metadata": {"ssf": float(current_ssf), "atr": atr}}
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"cutoff": self.cutoff, "a1": self.a1, "a2": self.a2, "b2": self.b2}
