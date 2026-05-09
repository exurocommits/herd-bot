import numpy as np

class NBarHighLowStrategy:
    """
    New 10-bar high = BUY. New 10-bar low = SELL. 
    Trail stop at previous bar's low (for longs) or high (for shorts). 
    Exit when trail hit. 
    Confidence: breakout magnitude / ATR.
    """
    def __init__(self, data):
        """
        data: numpy array or similar with columns [open, high, low, close, volume]
        """
        self.data = data
        self.n = 10
        self.atr_period = 14
        self.position = None  # 'LONG', 'SHORT', or None
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.signals = []
        self.current_idx = 0
        
        # Precompute ATR for confidence
        self.atr = self._calculate_atr(data, self.atr_period)

    def _calculate_atr(self, data, period):
        high = data[:, 1]
        low = data[:, 0] # Wait, data columns are [open, high, low, close, volume]
        # Correcting column mapping: 0:open, 1:high, 2:low, 3:close, 4:volume
        high = data[:, 1]
        low = data[:, 2]
        close = data[:, 3]
        
        tr = np.zeros_like(close)
        tr[1:] = np.maximum(high[1:] - low[1:], 
                            np.minimum(high[1:] - close[:-1], low[1:] - close[:-1]))
        
        atr = np.zeros_like(close)
        # Simple SMA for ATR for self-contained numpy
        for i in range(period, len(close)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        return atr

    def next(self, bar):
        """
        bar: a single row representing the current bar [open, high, low, close, volume]
        """
        # In a real backtest, we'd index into the precomputed data
        # But this interface receives 'bar' one by one. 
        # To make this robust, we should keep a rolling buffer of recent bars.
        # However, the instructions imply the class manages state.
        # Let's assume 'data' passed to init is the full historical array, 
        # and we use it to look back.
        
        # This is a simplified implementation assuming we are iterating through self.data
        # and 'bar' is the current row.
        pass

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"n": self.n, "atr_period": self.atr_period}

# Re-designing to work with the 'next(bar)' paradigm properly
class NBarHighLowStrategy:
    """
    New 10-bar high = BUY. New 10-bar low = SELL. 
    Trail stop at previous bar's low (for longs) or high (for shorts). 
    Exit when trail hit. 
    Confidence: breakout magnitude / ATR.
    """
    def __init__(self, data):
        self.data = data # Full historical data for lookback
        self.n = 10
        self.atr_period = 14
        self.position = None  # 'LONG', 'SHORT', or None
        self.entry_price = 0.0
        self.trail_stop = 0.0
        self.signals = []
        self.idx = 0
        
        # Precompute ATR
        self.atr = self._compute_atr()

    def _compute_atr(self):
        high = self.data[:, 1]
        low = self.data[:, 2]
        close = self.data[:, 3]
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        for i in range(self.atr_period, len(close)):
            atr[i] = np.mean(tr[i-self.atr_period:i])
        return atr

    def next(self, bar):
        # bar is current row. We need to know current index to look back.
        # Since the interface is next(bar), we track internal idx.
        curr_idx = self.idx
        if curr_idx < max(self.n, self.atr_period) + 1:
            self.idx += 1
            return {"type": "HOLD", "price": bar[3], "confidence": 0.0, "metadata": {}}

        highs = self.data[curr_idx-self.n:curr_idx, 1]
        lows = self.data[curr_idx-self.n:curr_idx, 2]
        
        current_high = bar[1]
        current_low = bar[2]
        current_close = bar[3]
        current_atr = self.atr[curr_idx]
        
        sig = {"type": "HOLD", "price": current_close, "confidence": 0.0, "metadata": {}}

        # Exit logic
        if self.position == 'LONG':
            if current_low <= self.trail_stop:
                self.signals.append({"type": "SELL", "price": current_close, "confidence": 1.0, "metadata": {"reason": "trail_hit"}})
                self.position = None
            else:
                # Update trail stop (previous bar's low)
                self.trail_stop = self.data[curr_idx-1, 2]
        elif self.position == 'SHORT':
            if current_high >= self.trail_stop:
                self.signals.append({"type": "BUY", "price": current_close, "confidence": 1.0, "metadata": {"reason": "trail_hit"}})
                self.position = None
            else:
                # Update trail stop (previous bar's high)
                self.trail_stop = self.data[curr_idx-1, 1]

        # Entry logic (only if not in position)
        if self.position is None:
            # Check for new high
            if current_high > np.max(highs):
                mag = current_high - np.max(highs)
                conf = mag / current_atr if current_atr > 0 else 0
                self.position = 'LONG'
                self.entry_price = current_close
                self.trail_stop = self.data[curr_idx, 2]
                sig = {"type": "BUY", "price": current_close, "confidence": conf, "metadata": {"breakout": "high"}}
                self.signals.append(sig)
            # Check for new low
            elif current_low < np.min(lows):
                mag = np.min(lows) - current_low
                conf = mag / current_atr if current_atr > 0 else 0
                self.position = 'SHORT'
                self.entry_price = current_close
                self.trail_stop = self.data[curr_idx, 1]
                sig = {"type": "SELL", "price": current_close, "confidence": conf, "metadata": {"breakout": "low"}}
                self.signals.append(sig)

        self.idx += 1
        return sig

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"n": self.n, "atr_period": self.atr_period}
