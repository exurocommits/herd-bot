import numpy as np

class AroonOscillator:
    """
    Aroon Oscillator Strategy.
    Aroon Up = (25 - periods_since_25_high) / 25 * 100.
    Aroon Down = (25 - periods_since_25_low) / 25 * 100.
    BUY when Aroon Up > 70 AND Aroon Down < 30.
    SELL opposite (Aroon Up < 30 AND Aroon Down > 70).
    Confidence: |Up - Down| / 100.
    """

    def __init__(self, data):
        """
        :param data: A numpy array of closing prices.
        """
        self.prices = data
        self.period = 25
        self.current_index = 0
        self._aroon_up, self._aroon_down = self._calculate_aroon(self.prices)

    def _calculate_aroon(self, data):
        n = len(data)
        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)
        
        for i in range(self.period, n):
            window = data[i - self.period : i + 1]
            
            # periods_since_high
            # Find index of max in window
            max_idx = np.argmax(window)
            # how many steps from the end of the window to that max_idx
            # window is size period + 1
            # if max is at the last element (i), periods_since = 0
            periods_since_high = self.period - max_idx
            
            # periods_since_low
            min_idx = np.argmin(window)
            periods_since_low = self.period - min_idx
            
            aroon_up[i] = ((self.period - periods_since_high) / self.period) * 100
            aroon_down[i] = ((self.period - periods_since_low) / self.period) * 100
            
        return aroon_up, aroon_down

    def next(self, bar):
        idx = self.current_index
        if idx >= len(self.prices):
            return {"type": "HOLD", "price": bar, "confidence": 0.0, "metadata": {}}

        price = bar
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}

        if not np.isnan(self._aroon_up[idx]) and not np.isnan(self._aroon_down[idx]):
            up = self._aroon_up[idx]
            down = self._aroon_down[idx]
            
            # BUY when Aroon Up > 70 AND Aroon Down < 30
            if up > 70 and down < 30:
                conf = min(abs(up - down) / 100.0, 1.0)
                signal = {"type": "BUY", "price": price, "confidence": conf, "metadata": {"up": up, "down": down}}
            
            # SELL opposite: Aroon Up < 30 AND Aroon Down > 70
            elif up < 30 and down > 70:
                conf = min(abs(up - down) / 100.0, 1.0)
                signal = {"type": "SELL", "price": price, "confidence": conf, "metadata": {"up": up, "down": down}}

        self.current_index += 1
        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"period": self.period}
