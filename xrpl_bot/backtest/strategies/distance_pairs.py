import numpy as np

class DistancePairs:
    """
    Distance Pairs strategy.
    Normalizes both price series (min-max over 250 bars).
    Distance is sum of squared differences between normalized series over 20 bars.
    Trades divergence/convergence.
    """
    def __init__(self, data):
        """
        :param data: dict with 'a' and 'b' as numpy arrays of prices.
        """
        self.price_a = data['a']
        self.price_b = data['b']
        self.norm_window = 250
        self.dist_window = 20
        self.position = 0
        self.current_idx = 0

    def _normalize(self, series, end_idx):
        start_idx = max(0, end_idx - self.norm_window + 1)
        window = series[start_idx : end_idx + 1]
        if len(window) < 2:
            return np.zeros_like(series[end_idx:end_idx+1])
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val == min_val:
            return np.zeros_like(window)
        return (window - min_val) / (max_val - min_val)

    def next(self, bar_idx):
        self.current_idx = bar_idx
        if bar_idx < self.norm_window + self.dist_window:
            return None

        # We need the normalized series for the last dist_window bars
        # To be efficient, we'll normalize the segment around the window
        # But the prompt says "Normalize both price series (min-max over 250 bars)"
        # This usually implies a rolling normalization.
        
        # Get the normalized values for the last 20 bars
        # To get accurate normalization, we need the 250-bar window preceding each of the 20 bars
        # However, for a single 'next' call, we can approximate by normalizing the last 250 bars
        # and then taking the last 20 of those.
        
        a_norm_segment = self._normalize(self.price_a, bar_idx)
        b_norm_segment = self._normalize(self.price_b, bar_idx)
        
        # Since _normalize returns only the window, we need to handle indexing.
        # Let's refine _normalize to return the window of the normalized values.
        # Actually, let's just compute the distance directly.
        
        # Get normalized windows
        # a_window_norm: last 20 bars, normalized using the 250 bars ending at bar_idx
        a_start = max(0, bar_idx - self.norm_window + 1)
        a_win_raw = self.price_a[a_start : bar_idx + 1]
        a_min, a_max = np.min(a_win_raw), np.max(a_win_raw)
        if a_max != a_min:
            a_win_norm = (a_win_raw - a_min) / (a_max - a_min)
        else:
            a_win_norm = np.zeros_like(a_win_raw)
            
        b_start = max(0, bar_idx - self.norm_window + 1)
        b_win_raw = self.price_b[b_start : bar_idx + 1]
        b_min, b_max = np.min(b_win_raw), np.max(b_win_raw)
        if b_max != b_min:
            b_win_norm = (b_win_raw - b_min) / (b_max - b_min)
        else:
            b_win_norm = np.zeros_like(b_win_raw)

        # Distance = sum((norm_a - norm_b)^2) over 20 bars? 
        # The prompt says "Distance = sum(...) over 20 bars". 
        # This usually means the distance metric is the sum of squared differences of the normalized series.
        # We'll compute this for the window.
        
        # We need the last 20 bars of the normalized 250-bar window.
        # Let's simplify: calculate normalized values for the last 250, then take last 20.
        
        # Re-calculating to be precise:
        def get_last_n_normalized(series, n_norm, n_dist, idx):
            start = max(0, idx - n_norm + 1)
            win = series[start : idx + 1]
            mi, ma = np.min(win), np.max(win)
            if ma == mi:
                norm_win = np.zeros_like(win)
            else:
                norm_win = (win - mi) / (ma - mi)
            return norm_win[-n_dist:]

        a_norm_20 = get_last_n_normalized(self.price_a, self.norm_window, self.dist_window, bar_idx)
        b_norm_20 = get_last_n_normalized(self.price_b, self.norm_window, self.dist_window, bar_idx)
        
        distance = np.sum((a_norm_20 - b_norm_20)**2)

        # For mean/std of distance, we need a history of distance.
        if not hasattr(self, 'dist_history'):
            self.dist_history = []
        self.dist_history.append(distance)
        if len(self.dist_history) > 100: # history buffer
            self.dist_history.pop(0)
            
        if len(self.dist_history) < 30:
            return None
            
        mean_dist = np.mean(self.dist_history)
        std_dist = np.std(self.dist_history)
        
        signal = {"type": "HOLD", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"distance": distance}}

        if self.position == 0:
            # BUY when distance crosses above 2*std from mean (divergence)
            if distance > mean_dist + 2 * std_dist:
                self.position = 1
                conf = (distance - mean_dist) / (2 * std_dist) if std_dist != 0 else 0
                signal = {"type": "BUY", "price": self.price_a[bar_idx], "confidence": min(conf, 1.0), "metadata": {"distance": distance}}
        elif self.position == 1:
            # SELL when distance crosses back below mean
            if distance < mean_dist:
                self.position = 0
                signal = {"type": "SELL", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"distance": distance}}

        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"norm_window": 250, "dist_window": 20}
