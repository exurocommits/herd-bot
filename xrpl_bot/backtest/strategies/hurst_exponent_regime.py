import numpy as np

class HurstExponentRegime:
    """
    Rolling Hurst exponent via R/S analysis over 100-bar window.
    H < 0.45 = mean-reverting regime -> BUY oversold (RSI), SELL overbought (RSI).
    H > 0.55 = trending regime -> follow trend (EMA).
    H between = HOLD.
    Confidence: |H - 0.5| / 0.5.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close', 'high', 'low' as numpy arrays.
        """
        self.close = np.array(data['close'], dtype=float)
        self.high = np.array(data['high'], dtype=float)
        self.low = np.array(data['low'], dtype=float)
        self.window = 100
        self.rsi_period = 14
        self.ema_period = 20
        self.signals = []
        self.current_idx = len(self.close) - 1

    def _get_hurst(self, series):
        n = len(series)
        if n < 10: return 0.5
        try:
            mean_adj = series - np.mean(series)
            cum_sum = np.cumsum(mean_adj)
            r = np.max(cum_sum) - np.min(cum_sum)
            s = np.std(series)
            if s == 0: return 0.5
            rs = r / s
            # H ~ log(R/S) / log(n)
            h = np.log(rs) / np.log(n) if rs > 0 and n > 1 else 0.5
            return float(np.clip(h, 0, 1))
        except:
            return 0.5

    def _rsi(self, series, period):
        if len(series) < period + 1:
            return 50.0
        subset = series[-(period+1):]
        deltas = np.diff(subset)
        up = deltas[deltas >= 0].sum()
        down = -deltas[deltas < 0].sum()
        if down == 0: return 100.0
        if up == 0: return 0.0
        rs = (up/period) / (down/period)
        return 100.0 - (100.0 / (1.0 + rs))

    def _ema(self, series, period):
        if len(series) < period:
            return series[-1]
        alpha = 2 / (period + 1)
        ema = series[0]
        for val in series[1:]:
            ema = (val * alpha) + (ema * (1 - alpha))
        return ema

    def next(self, bar):
        """
        :param bar: dict containing 'close', 'high', 'low'
        """
        # Assuming next(bar) is called for each new bar.
        # We update our internal state.
        self.close = np.append(self.close, bar['close'])
        self.high = np.append(self.high, bar['high'])
        self.low = np.append(self.low, bar['low'])
        self.current_idx += 1
        
        # Need enough data for the window
        if self.current_idx < self.window:
            return {"type": "HOLD", "price": float(bar['close']), "confidence": 0.0, "metadata": {}}

        window_close = self.close[-self.window:]
        h = self._get_hurst(window_close)
        price = float(bar['close'])
        conf = float(np.clip(abs(h - 0.5) / 0.5, 0, 1))
        
        signal_type = "HOLD"
        if h < 0.45: # Mean Reverting
            rsi = self._rsi(window_close, self.rsi_period)
            if rsi < 30: signal_type = "BUY"
            elif rsi > 70: signal_type = "SELL"
        elif h > 0.55: # Trending
            ema = self._ema(window_close, self.ema_period)
            if price > ema: signal_type = "BUY"
            elif price < ema: signal_type = "SELL"
            
        signal = {"type": signal_type, "price": price, "confidence": conf, "metadata": {"hurst": h}}
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"window": self.window, "rsi_period": self.rsi_period, "ema_period": self.ema_period}
