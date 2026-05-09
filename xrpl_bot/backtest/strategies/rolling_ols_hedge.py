import numpy as np

class RollingOLSHedge:
    """
    Rolling OLS Hedge strategy.
    Regresses price_a on price_b over a rolling window to find hedge ratio beta.
    Trades the spread: price_a - beta * price_b.
    """
    def __init__(self, data):
        """
        :param data: dict with 'a' and 'b' as numpy arrays of prices.
        """
        self.price_a = data['a']
        self.price_b = data['b']
        self.window = 100
        self.current_idx = 0
        self.position = 0  # 1 for long spread, -1 for short spread, 0 for flat
        self.history = []

    def next(self, bar_idx):
        """
        Processes the next bar.
        :param bar_idx: Current index in the data series.
        :return: dict signal or None.
        """
        self.current_idx = bar_idx
        if bar_idx < self.window:
            return None

        # Get windows
        a_win = self.price_a[bar_idx - self.window + 1 : bar_idx + 1]
        b_win = self.price_b[bar_idx - self.window + 1 : bar_idx + 1]

        # Rolling OLS: price_a = beta * price_b + alpha
        # Using numpy polyfit for simplicity (degree 1)
        beta, alpha = np.polyfit(b_win, a_win, 1)

        # Current spread
        current_spread = self.price_a[bar_idx] - (beta * self.price_b[bar_idx] + alpha)
        
        # Calculate historical spreads in window for Z-score
        # Note: To be strictly rolling, we should calculate spread for each point in the window
        # However, a common approximation is to use the current beta for the window
        spreads = a_win - (beta * b_win + alpha)
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        if std_spread == 0:
            z_score = 0
        else:
            z_score = (current_spread - mean_spread) / std_spread

        signal = {"type": "HOLD", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"z": z_score, "beta": beta}}

        # Strategy Logic
        if self.position == 0:
            if z_score < -2:
                self.position = 1
                signal = {"type": "BUY", "price": self.price_a[bar_idx], "confidence": min(abs(z_score) / 3.0, 1.0), "metadata": {"z": z_score, "beta": beta}}
            elif z_score > 2:
                self.position = -1
                signal = {"type": "SELL", "price": self.price_a[bar_idx], "confidence": min(abs(z_score) / 3.0, 1.0), "metadata": {"z": z_score, "beta": beta}}
        
        elif self.position == 1: # Long spread
            if z_score >= 0:
                self.position = 0
                signal = {"type": "SELL", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"z": z_score, "beta": beta}}
        
        elif self.position == -1: # Short spread
            if z_score <= 0:
                self.position = 0
                signal = {"type": "BUY", "price": self.price_a[bar_idx], "confidence": 0.0, "metadata": {"z": z_score, "beta": beta}}

        return signal

    def get_signals(self):
        """Returns the history of signals (simplified for this implementation)."""
        return self.history

    def get_params(self):
        """Returns strategy parameters."""
        return {"window": self.window}
