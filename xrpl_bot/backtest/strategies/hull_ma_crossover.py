import numpy as np

class HullMACrossover:
    """
    Hull Moving Average (HMA) Crossover Strategy.
    HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n)).
    Uses periods 9 and 52.
    BUY when fast HMA crosses above slow HMA.
    SELL when fast HMA crosses below slow HMA.
    Confidence: distance / ATR.
    """

    def __init__(self, data):
        """
        :param data: A numpy array of closing prices.
        """
        self.prices = data
        self.fast_period = 9
        self.slow_period = 52
        self.atr_period = 14
        self.signals = []
        self.current_index = 0
        self._hma_fast = self._calculate_hma(self.prices, self.fast_period)
        self._hma_slow = self._calculate_hma(self.prices, self.slow_period)
        self._atr = self._calculate_atr(self.prices, self.atr_period)

    def _wma(self, data, period):
        if len(data) < period:
            return np.array([np.nan] * len(data))
        weights = np.arange(1, period + 1)
        wma = np.zeros_like(data)
        wma[:] = np.nan
        for i in range(period - 1, len(data)):
            slice_data = data[i - period + 1 : i + 1]
            wma[i] = np.sum(slice_data * weights) / weights.sum()
        return wma

    def _calculate_hma(self, data, period):
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = self._wma(data, half_period)
        wma_full = self._wma(data, period)
        
        diff = 2 * wma_half - wma_full
        # We need to apply WMA to the diff, but WMA needs a non-nan input.
        # Since diff has NaNs at the beginning, we'll handle it.
        hma = np.full_like(data, np.nan)
        
        # Find the first index where diff is not NaN
        first_valid = np.where(~np.isnan(diff))[0]
        if len(first_valid) == 0:
            return hma
            
        start_idx = first_valid[0]
        # We need enough data from start_idx to compute WMA(sqrt_period)
        # But WMA(sqrt_period) needs sqrt_period valid points.
        # To be safe and simple, we'll compute it on the whole diff array.
        # However, the 'diff' array might have NaNs. We'll fill them with 0 for the WMA calculation
        # but that's mathematically incorrect. Better to just slide over the valid part.
        
        for i in range(start_idx + sqrt_period - 1, len(data)):
            slice_diff = diff[i - sqrt_period + 1 : i + 1]
            # If any NaN in slice, skip
            if np.isnan(slice_diff).any():
                continue
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.sum(slice_diff * weights) / weights.sum()
            
        return hma

    def _calculate_atr(self, data, period):
        if len(data) < period + 1:
            return np.array([np.nan] * len(data))
        
        high_low = np.abs(data[1:] - data[:-1]) # Simplified as we only have close
        # In a real backtest, we'd use high/low/close. Here we'll use close as proxy.
        # Since we only have one data stream (prices), we use abs diff.
        tr = high_low
        atr = np.zeros_like(data)
        atr[:] = np.nan
        
        for i in range(period, len(data)):
            atr[i] = np.mean(tr[i-period:i])
        return atr

    def next(self, bar):
        """
        :param bar: Current price (float).
        :return: dict signal
        """
        idx = self.current_index
        if idx >= len(self.prices):
            return {"type": "HOLD", "price": bar, "confidence": 0.0, "metadata": {}}

        price = bar
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}

        if idx > 0 and idx < len(self._hma_fast) and idx < len(self._hma_slow):
            prev_idx = idx - 1
            
            fast_now = self._hma_fast[idx]
            fast_prev = self._hma_fast[prev_idx]
            slow_now = self._hma_slow[idx]
            slow_prev = self._hma_slow[prev_idx]
            
            atr_val = self._atr[idx]
            if np.isnan(atr_val) or atr_val == 0:
                atr_val = 1.0 # Avoid division by zero
                
            if not np.isnan(fast_now) and not np.isnan(slow_now) and \
               not np.isnan(fast_prev) and not np.isnan(slow_prev):
                
                # BUY: Fast crosses above Slow
                if fast_prev <= slow_prev and fast_now > slow_now:
                    dist = abs(fast_now - slow_now)
                    confidence = min(dist / atr_val, 1.0)
                    signal = {"type": "BUY", "price": price, "confidence": confidence, "metadata": {"fast": fast_now, "slow": slow_now}}
                
                # SELL: Fast crosses below Slow
                elif fast_prev >= slow_prev and fast_now < slow_now:
                    dist = abs(fast_now - slow_now)
                    confidence = min(dist / atr_val, 1.0)
                    signal = {"type": "SELL", "price": price, "confidence": confidence, "metadata": {"fast": fast_now, "slow": slow_now}}

        self.current_index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}
