import numpy as np

class KAMAAdaptive:
    """
    Kaufman Adaptive Moving Average (KAMA) Strategy.
    Efficiency ratio = |direction| / volatility.
    Smoothing constant adapts.
    BUY when close > KAMA and KAMA rising.
    SELL when close < KAMA and KAMA falling.
    Period=10, fast=2, slow=30.
    """

    def __init__(self, data):
        """
        :param data: A numpy array of closing prices.
        """
        self.prices = data
        self.period = 10
        self.fast_period = 2
        self.slow_period = 30
        self.current_index = 0
        self._kama = self._calculate_kama(self.prices)

    def _calculate_kama(self, data):
        n = len(data)
        kama = np.full(n, np.nan)
        
        if n <= self.period:
            return kama

        # Pre-calculate components
        # Direction: total change over n periods
        # Volatility: sum of absolute price changes over n periods
        
        # For efficiency, we'll compute it iteratively in the 'next' context if needed,
        # but for the class to be self-contained and fast, let's pre-calculate.
        
        # Note: In a real real-time system, we'd do this incrementally.
        # For backtesting, pre-calculation is fine.
        
        # Initial KAMA is the SMA of the first 'period'
        # However, standard KAMA uses a smoothing constant.
        
        # Pre-calculate absolute changes for volatility
        abs_diffs = np.abs(np.diff(data))
        
        # We'll iterate to compute KAMA
        # KAMA[i] = KAMA[i-1] + SC * (Price[i] - KAMA[i-1])
        # SC = [ER * (fast_const - slow_const) + slow_const]^2
        
        fast_const = 2 / (self.fast_period + 1)
        slow_const = 2 / (self.slow_period + 1)
        
        # Seed the first KAMA with SMA
        # But we need to wait until we have 'period' data for ER.
        # Let's start from 'period' index.
        
        # We'll use a simple loop for the KAMA calculation.
        # This is slow for large arrays in pure Python but okay for typical strategy use.
        
        current_kama = np.mean(data[:self.period])
        kama[self.period - 1] = current_kama
        
        for i in range(self.period, n):
            # Efficiency Ratio (ER)
            direction = abs(data[i] - data[i - self.period])
            volatility = np.sum(abs_diffs[i - self.period : i])
            
            if volatility == 0:
                er = 0
            else:
                er = direction / volatility
                
            sc = (er * (fast_const - slow_const) + slow_const) ** 2
            current_kama = current_kama + sc * (data[i] - current_kama)
            kama[i] = current_kama
            
        return kama

    def next(self, bar):
        idx = self.current_index
        if idx >= len(self.prices):
            return {"type": "HOLD", "price": bar, "confidence": 0.0, "metadata": {}}

        price = bar
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}

        if idx > 0 and not np.isnan(self._kama[idx]) and not np.isnan(self._kama[idx-1]):
            kama_now = self._kama[idx]
            kama_prev = self._kama[idx-1]
            
            # BUY when close > KAMA and KAMA rising
            if price > kama_now and kama_now > kama_prev:
                signal = {"type": "BUY", "price": price, "confidence": 1.0, "metadata": {"kama": kama_now}}
            
            # SELL when close < KAMA and KAMA falling
            elif price < kama_now and kama_now < kama_prev:
                signal = {"type": "SELL", "price": price, "confidence": 1.0, "metadata": {"kama": kama_now}}

        self.current_index += 1
        return signal

    def get_signals(self):
        # This is a placeholder as the class is designed for streaming/next-based usage
        return []

    def get_params(self):
        return {"period": self.period, "fast": self.fast_period, "slow": self.slow_period}
