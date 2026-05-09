import numpy as np

class VixCorrelationProxyStrategy:
    """
    VIX Correlation Proxy Strategy.
    Use BTC 10-day realized volatility as VIX proxy.
    high_vol = vol > 75th percentile of 100-bar history.
    low_vol = vol < 25th percentile.
    high_vol regime: mean-reversion strategies (BUY oversold RSI<30, SELL overbought RSI>70).
    low_vol regime: breakout strategies (BUY on 20-bar high).
    Confidence: |percentile - 50| / 50.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close' as numpy array.
        """
        self.data = data
        self.close = data['close']
        self.window_vol = 10
        self.window_percentile = 100
        self.window_rsi = 14
        self.window_breakout = 20
        self.signals = []

    def _rsi(self, arr, window):
        deltas = np.diff(arr)
        seed = np.zeros(len(arr))
        # This is a simple RSI implementation
        # In a real scenario, we'd use a proper rolling RSI
        # But for this self-contained class:
        res = np.zeros(len(arr))
        for i in range(window, len(arr)):
            diffs = arr[i-window:i+1]
            gain = np.where(np.diff(diffs) > 0, np.diff(diffs), 0)
            loss = np.where(np.diff(diffs) < 0, -np.diff(diffs), 0)
            avg_gain = np.mean(gain)
            avg_loss = np.mean(loss)
            if avg_loss == 0:
                res[i] = 100
            else:
                rs = avg_gain / avg_loss
                res[i] = 100 - (100 / (1 + rs))
        return res

    def _realized_vol(self, arr, window):
        # Simplified realized vol: std of log returns
        log_returns = np.diff(np.log(arr), prepend=np.log(arr[0]))
        vols = np.zeros(len(arr))
        for i in range(window, len(arr)):
            vols[i] = np.std(log_returns[i-window:i+1])
        return vols

    def next(self, bar_idx):
        if bar_idx < max(self.window_percentile, self.window_breakout, self.window_rsi):
            return {"type": "HOLD", "price": float(self.close[bar_idx]), "confidence": 0.0, "metadata": {}}

        # 1. Calculate 10-day realized vol
        # We need the whole series to calculate vols up to bar_idx
        vols = self._realized_vol(self.close, self.window_vol)
        current_vol = vols[bar_idx]
        
        # 2. Determine percentile relative to 100-bar history
        vol_history = vols[max(0, bar_idx - self.window_percentile):bar_idx+1]
        # Use percentile to find position
        # percentile = (count(x < current) / total) * 100
        percentile = (np.sum(vol_history < current_vol) / len(vol_history)) * 100
        
        # 3. Regime identification
        high_vol_regime = percentile > 75
        low_vol_regime = percentile < 25
        
        signal_type = "HOLD"
        confidence = abs(percentile - 50) / 50
        
        # RSI for mean reversion
        rsi_vals = self._rsi(self.close, self.window_rsi)
        current_rsi = rsi_vals[bar_idx]
        
        # Breakout for low vol
        breakout_high = np.max(self.close[bar_idx - self.window_breakout : bar_idx])

        if high_vol_regime:
            # Mean reversion
            if current_rsi < 30:
                signal_type = "BUY"
            elif current_rsi > 70:
                signal_type = "SELL"
        elif low_vol_regime:
            # Breakout
            if self.close[bar_idx] > breakout_high:
                signal_type = "BUY"

        return {
            "type": signal_type,
            "price": float(self.close[bar_idx]),
            "confidence": float(min(1.0, confidence)),
            "metadata": {
                "vol": float(current_vol),
                "percentile": float(percentile),
                "rsi": float(current_rsi)
            }
        }

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {
            "vol_window": self.window_vol,
            "percentile_window": self.window_percentile,
            "rsi_window": self.window_rsi,
            "breakout_window": self.window_breakout
        }
