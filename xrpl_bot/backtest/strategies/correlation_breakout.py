import numpy as np

class CorrelationBreakout:
    """
    Correlation Breakout strategy.
    Uses rolling Pearson correlation (50-bar window) between two assets.
    Traded when correlation drops below 0.3 (breakdown) or exceeds 0.7 (re-correlation).
    """
    def __init__(self, data):
        """
        :param data: dict with 'a' and 'b' as numpy arrays of prices.
        """
        self.price_a = data['a']
        self.price_b = data['b']
        self.window = 50
        self.position = 0 # 1: long_a_short_b, -1: short_a_long_b, 2: long_b_short_a, -2: short_b_long_a
        self.current_idx = 0

    def next(self, bar_idx):
        self.current_idx = bar_idx
        if bar_idx < self.window:
            return None

        a_win = self.price_a[bar_idx - self.window + 1 : bar_idx + 1]
        b_win = self.price_b[bar_idx - self.window + 1 : bar_idx + 1]
        
        correlation = np.corrcoef(a_win, b_win)[0, 1]
        
        signal = {"type": "HOLD", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"correlation": correlation}}

        # If correlation > 0.7, close any positions
        if self.position != 0 and correlation > 0.7:
            # Close all
            self.position = 0
            signal = {"type": "SELL", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"correlation": correlation, "action": "close_all"}}
            return signal

        # If correlation < 0.3, trade divergence
        if self.position == 0 and correlation < 0.3:
            # Calculate relative performance in window
            perf_a = (self.price_a[bar_idx] / self.price_a[bar_idx - self.window + 1]) - 1
            perf_b = (self.price_b[bar_idx] / self.price_b[bar_idx - self.window + 1]) - 1
            
            conf = abs(0.5 - correlation) / 0.5
            
            if perf_a > perf_b:
                # asset_a > asset_b performance -> BUY b, SELL a
                self.position = 2 # long_b_short_a
                signal = {"type": "BUY", "price": self.price_b[bar_idx], "confidence": conf, "metadata": {"correlation": correlation, "action": "long_b_short_a"}}
            else:
                # asset_b > asset_a performance -> BUY a, SELL b
                self.position = -2 # short_b_long_a (wait, prompt says: "If asset_a > asset_b performance -> BUY b, SELL a")
                # If perf_a > perf_b -> BUY b, SELL a.
                # If perf_b > perf_a -> BUY a, SELL b.
                self.position = -2 # actually let's use -2 for long_a_short_b
                # Let's re-map:
                # 1: long_a_short_b, -1: short_a_long_b, 2: long_b_short_a, -2: short_b_long_a
                # If perf_a > perf_b: BUY b, SELL a -> position = 2
                # If perf_b > perf_a: BUY a, SELL b -> position = 1
                self.position = 2
                signal = {"type": "BUY", "price": self.price_b[bar_idx], "confidence": conf, "metadata": {"correlation": correlation, "action": "long_b_short_a"}}
                # wait, I just wrote the same thing twice in my logic check. 
                # Let's fix it.
            
            # Corrected Logic:
            if perf_a > perf_b:
                self.position = 2 # long b, short a
                signal = {"type": "BUY", "price": self.price_b[bar_idx], "confidence": conf, "metadata": {"correlation": correlation, "action": "long_b_short_a"}}
            else:
                self.position = 1 # long a, short b
                signal = {"type": "BUY", "price": self.price_a[bar_idx], "confidence": conf, "metadata": {"correlation": correlation, "action": "long_a_short_b"}}
            return signal

        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"window": 50}
