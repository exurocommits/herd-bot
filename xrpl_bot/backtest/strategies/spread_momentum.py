import numpy as np

class SpreadMomentum:
    """
    Spread Momentum strategy.
    Calculates log spread = log(price_a) - log(price_b).
    Trades momentum of the spread.
    """
    def __init__(self, data):
        """
        :param data: dict with 'a' and 'b' as numpy arrays of prices.
        """
        self.price_a = data['a']
        self.price_b = data['b']
        self.log_a = np.log(self.price_a)
        self.log_b = np.log(self.price_b)
        self.spread = self.log_a - self.log_b
        
        self.roc_window = 10
        self.position = 0 # 1: long spread (long a, short b), -1: short spread (short a, long b)
        self.current_idx = 0

    def next(self, bar_idx):
        self.current_idx = bar_idx
        if bar_idx < self.roc_window + 5:
            return None

        # Spread ROC over 10 bars
        # ROC = (spread[t] - spread[t-10]) / spread[t-10] (or just diff)
        # Prompt says: "Spread ROC over 10 bars > 0 and accelerating"
        # Let's use the difference as ROC for simplicity in a discrete setting
        
        spread_window = self.spread[bar_idx - self.roc_window : bar_idx + 1]
        roc = spread_window[-1] - spread_window[0]
        
        # To check "accelerating", we need the previous ROC
        prev_roc_window = self.spread[bar_idx - 2*self.roc_window : bar_idx - self.roc_window + 1]
        if len(prev_roc_window) < self.roc_window:
            return None
        prev_roc = prev_roc_window[-1] - prev_roc_window[0]
        
        # Standard deviation of ROC for confidence
        # We'll use a rolling window of ROCs
        roc_history = []
        for i in range(max(0, bar_idx - 50), bar_idx + 1):
            if i < self.roc_window: continue
            s_win = self.spread[i - self.roc_window : i + 1]
            roc_history.append(s_win[-1] - s_win[0])
        
        std_roc = np.std(roc_history) if len(roc_history) > 1 else 1.0
        conf = abs(roc) / std_roc if std_roc != 0 else 0

        signal = {"type": "HOLD", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"roc": roc}}

        # Logic:
        # Spread ROC > 0 and accelerating -> BUY spread (long a, short b)
        # Spread ROC < 0 and decelerating -> SELL (Wait, if ROC < 0 and accelerating (getting more negative) -> SELL)
        # Let's clarify:
        # BUY: roc > 0 and roc > prev_roc
        # SELL: roc < 0 and roc < prev_roc
        
        if self.position == 0:
            if roc > 0 and roc > prev_roc:
                self.position = 1
                signal = {"type": "BUY", "price": self.price_a[bar_idx], "confidence": min(conf, 1.0), "metadata": {"roc": roc}}
            elif roc < 0 and roc < prev_roc:
                self.position = -1
                signal = {"type": "SELL", "price": self.price_a[bar_idx], "confidence": min(conf, 1.0), "metadata": {"roc": roc}}
        
        elif self.position == 1: # Long spread
            # Exit on momentum reversal (roc turns negative or starts decelerating?)
            # Prompt: "Exit on momentum reversal"
            if roc < 0 or roc < prev_roc:
                self.position = 0
                signal = {"type": "SELL", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"roc": roc}}
                
        elif self.position == -1: # Short spread
            if roc > 0 or roc > prev_roc:
                self.position = 0
                signal = {"type": "BUY", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"roc": roc}}

        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"roc_window": 10}
