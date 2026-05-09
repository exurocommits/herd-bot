import numpy as np

class PivotPointBounce:
    """
    Classic pivot = (H+L+C)/3.
    S1 = 2*P - H, R1 = 2*P - L.
    S2 = P - (H-L), R2 = P + (H-L).
    BUY at S1 (support bounce confirmed by close > S1).
    SELL at R1 (resistance confirmed).
    Stop at S2/R2.
    Confidence: distance from pivot / (R1 - S1).
    """
    def __init__(self, data):
        self.high = np.array(data['high'], dtype=float)
        self.low = np.array(data['low'], dtype=float)
        self.close = np.array(data['close'], dtype=float)
        self.signals = []
        self.current_idx = len(self.close) - 1
        self.in_position = False
        self.position_type = None # "BUY" or "SELL"

    def next(self, bar):
        # We need the PREVIOUS bar's H, L, C to calculate P, S1, R1, S2, R2
        # Since the data is appended, we'll use the data up to current_idx
        self.high = np.append(self.high, bar['high'])
        self.low = np.append(self.low, bar['low'])
        self.close = np.append(self.close, bar['close'])
        self.current_idx += 1

        if self.current_idx < 1:
            return {"type": "HOLD", "price": float(bar['close']), "confidence": 0.0, "metadata": {}}

        # Pivot based on PREVIOUS bar
        prev_h = self.high[self.current_idx - 1]
        prev_l = self.low[self.current_idx - 1]
        prev_c = self.close[self.current_idx - 1]

        p = (prev_h + prev_l + prev_c) / 3
        s1 = 2 * p - prev_h
        r1 = 2 * p - prev_l
        s2 = p - (prev_h - prev_l)
        r2 = p + (prev_h - prev_l)

        price = float(bar['close'])
        signal_type = "HOLD"
        conf = 0.0
        
        # Confidence calculation
        denom = (r1 - s1)
        if denom != 0:
            conf = float(np.clip(abs(price - p) / denom, 0, 1))
        else:
            conf = 0.0

        if not self.in_position:
            # BUY at S1 (support bounce confirmed by close > S1)
            if price > s1 and self.low[self.current_idx] <= s1: # Simplified: low touched/passed s1 and close > s1
                 # Actually, prompt says "BUY at S1 (support bounce confirmed by close > S1)"
                 # Let's interpret: if close > s1 and price was at/below s1 in this bar.
                 if price > s1 and bar['low'] <= s1:
                     signal_type = "BUY"
                     self.in_position = True
                     self.position_type = "BUY"
            
            # SELL at R1 (resistance confirmed)
            elif price < r1 and bar['high'] >= r1:
                signal_type = "SELL"
                self.in_position = True
                self.position_type = "SELL"
        else:
            # Stop loss or profit taking
            if self.position_type == "BUY":
                if price <= s2 or price >= r1: # Stop at S2 or exit at R1
                    signal_type = "SELL" # Exit
                    self.in_position = False
                    self.position_type = None
                elif price < s1: # Just a bounce check? The prompt is a bit ambiguous on "SELL at R1" 
                                # for a BUY position. Usually, BUY exits at R1.
                    pass 
            elif self.position_type == "SELL":
                if price >= r2 or price <= s1: # Stop at R2 or exit at S1
                    signal_type = "BUY" # Cover
                    self.in_position = False
                    self.position_type = None

        signal = {"type": signal_type, "price": price, "confidence": conf, "metadata": {"p": p, "s1": s1, "r1": r1, "s2": s2, "r2": r2}}
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {}
