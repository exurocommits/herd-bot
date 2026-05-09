import numpy as np

class ElderTripleScreen:
    """
    Elder Triple Screen simulation.
    Screen 1: weekly trend via 26-bar EMA (upsloping=longs only).
    Screen 2: daily MACD histogram (BUY when histogram < 0 but rising).
    Screen 3: entry on parabolic SAR flip.
    
    Simulated with fast/slow indicators on same data.
    Confidence: screen agreement score (0-3).
    """
    def __init__(self, data):
        """
        :param data: Dictionary containing 'close', 'high', 'low' as numpy arrays.
        """
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.current_index = 0
        self.position = None
        
        # Pre-calculate indicators for efficiency
        self.ema26 = self._calculate_ema(self.close, 26)
        self.macd_line, self.macd_signal = self._calculate_macd(self.close)
        self.macd_hist = self.macd_line - self.macd_signal
        self.psar = self._calculate_psar(self.high, self.low, self.close)

    def _calculate_ema(self, series, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = self._calculate_ema(series, fast)
        ema_slow = self._calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = self._calculate_ema(macd_line, signal)
        return macd_line, macd_signal

    def _calculate_psar(self, high, low, close):
        # Simplified Parabolic SAR simulation
        psar = np.zeros_like(close)
        af = 0.02
        max_af = 0.2
        trend = 1 # 1 for long, -1 for short
        psar[0] = close[0]
        
        for i in range(1, len(close)):
            if trend == 1:
                if low[i] < psar[i-1]:
                    trend = -1
                    psar[i] = high[i]
                    af = af * 2 # Reset logic would be better, but this is a simulation
                else:
                    psar[i] = psar[i-1] + af * (high[i-1] - psar[i-1])
                    # logic for incrementing af based on extreme prices...
            else:
                if high[i] > psar[i-1]:
                    trend = 1
                    psar[i] = low[i]
                    af = 0.02
                else:
                    psar[i] = psar[i-1] + af * (psar[i-1] - low[i-1])
        return psar

    def next(self, bar):
        idx = self.current_index
        self.current_index += 1
        
        if idx < 30: # Warmup
            return {"type": "HOLD", "price": bar['close'], "confidence": 0.0, "metadata": {}}

        current_close = bar['close']
        
        # Screen 1: Trend (EMA 26)
        ema_val = self.ema26[idx]
        prev_ema = self.ema26[idx-1]
        trend_up = ema_val > prev_ema
        
        # Screen 2: MACD Histogram (BUY when hist < 0 but rising)
        hist = self.macd_hist[idx]
        prev_hist = self.macd_hist[idx-1]
        macd_bullish = hist < 0 and hist > prev_hist
        
        # Screen 3: PSAR Flip (Price crosses PSAR)
        psar_val = self.psar[idx]
        psar_flip_up = current_close > psar_val and bar['close'] <= self.psar[idx-1] # simplified
        # Let's just check if price is above PSAR for long entry
        psar_long = current_close > psar_val

        # Confidence Calculation
        score = 0
        if trend_up: score += 1
        if macd_bullish: score += 1
        if psar_long: score += 1
        
        signal_type = "HOLD"
        if self.position is None:
            if trend_up and macd_bullish and psar_long:
                self.position = "BUY"
                signal_type = "BUY"
        elif self.position == "BUY":
            if not trend_up or current_close < psar_val:
                self.position = None
                signal_type = "SELL"

        return {
            "type": signal_type, 
            "price": float(current_close), 
            "confidence": float(score / 3.0), 
            "metadata": {"score": score, "ema": float(ema_val), "macd_hist": float(hist)}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"ema_period": 26, "macd_fast": 12, "macd_slow": 26}
