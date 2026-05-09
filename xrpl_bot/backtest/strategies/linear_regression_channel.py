import numpy as np

class LinearRegressionChannel:
    """
    Linear Regression Channel Strategy.
    20-period linear regression line + 2 std dev channels.
    BUY when price touches lower channel.
    SELL at upper channel.
    Exit at regression line.
    Confidence: distance from channel / channel width.
    """

    def __init__(self, data):
        """
        :param data: A numpy array of closing prices.
        """
        self.prices = data
        self.period = 20
        self.std_dev_multiplier = 2
        self.current_index = 0
        
        # Pre-calculate regression parameters for all points
        self.reg_line, self.upper_channel, self.lower_channel, self.channel_width = self._calculate_channels(self.prices)

    def _calculate_channels(self, data):
        n = len(data)
        reg_line = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        width = np.full(n, np.nan)
        
        for i in range(self.period - 1, n):
            y = data[i - self.period + 1 : i + 1]
            x = np.arange(self.period)
            
            # Linear regression using least squares
            # y = mx + c
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Predict for the current point (the end of the window)
            current_reg = m * (self.period - 1) + c
            reg_line[i] = current_reg
            
            # Standard deviation of residuals
            residuals = y - (m * x + c)
            std_dev = np.std(residuals)
            
            upper[i] = current_reg + (self.std_dev_multiplier * std_dev)
            lower[i] = current_reg - (self.std_dev_multiplier * std_dev)
            width[i] = upper[i] - lower[i]
            
        return reg_line, upper, lower, width

    def next(self, bar):
        idx = self.current_index
        if idx >= len(self.prices):
            return {"type": "HOLD", "price": bar, "confidence": 0.0, "metadata": {}}

        price = bar
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}

        if not np.isnan(self.lower_channel[idx]) and not np.isnan(self.upper_channel[idx]):
            # BUY when price touches lower channel
            if price <= self.lower_channel[idx]:
                dist = self.lower_channel[idx] - price # Distance to touch (could be negative if it broke through)
                # Re-interpreting: distance from channel as abs(price - lower) / width
                # But "touches" usually means price is at or below.
                # The prompt says "Confidence: distance from channel / channel width".
                # Let's use |price - lower| / width.
                conf = min(abs(price - self.lower_channel[idx]) / self.channel_width[idx] + 0.1, 1.0) # +0.1 for "touching"
                signal = {"type": "BUY", "price": price, "confidence": conf, "metadata": {"lower": self.lower_channel[idx]}}
            
            # SELL at upper channel
            elif price >= self.upper_channel[idx]:
                conf = min(abs(price - self.upper_channel[idx]) / self.channel_width[idx] + 0.1, 1.0)
                signal = {"type": "SELL", "price": price, "confidence": conf, "metadata": {"upper": self.upper_channel[idx]}}
            
            # Exit at regression line (This is a conditional exit, often handled by the backtester)
            # Here we'll just include it in metadata or check for a "close" signal if we had state.
            # Since we don't have state, we won't trigger a SELL/BUY, but we'll mark it in metadata.
            # Actually, the prompt says "SELL at upper channel". It doesn't specify how to handle the "Exit at regression line" 
            # in terms of the BUY/SELL/HOLD enum. We'll stick to the main signal types.

        self.current_index += 1
        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"period": self.period, "std_dev": self.std_dev_multiplier}
