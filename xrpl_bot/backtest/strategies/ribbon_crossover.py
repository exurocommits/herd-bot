import numpy as np

class RibbonCrossover:
    """
    8 EMAs from period 10 to 80 (10,20,30,40,50,60,70,80).
    BUY when all EMAs align upward (each > previous).
    SELL when all compress (std of EMAs / mean < threshold).
    Confidence: ribbon spread (max EMA - min EMA) / price.
    """
    def __init__(self, data):
        """
        :param data: Dictionary containing 'close' as numpy array.
        """
        self.close = data['close']
        self.periods = [10, 20, 30, 40, 50, 60, 70, 80]
        self.current_index = 0
        self.position = None
        self.threshold = 0.001 # Compression threshold
        
        # Pre-calculate all EMAs
        self.emas = {}
        for p in self.periods:
            self.emas[p] = self._calculate_ema(self.close, p)

    def _calculate_ema(self, series, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

    def next(self, bar):
        idx = self.current_index
        self.current_index += 1
        
        current_close = bar['close']
        if idx < 80:
            return {"type": "HOLD", "price": current_close, "confidence": 0.0, "metadata": {}}

        # Get EMA values for current and previous bar to check alignment
        current_ema_vals = np.array([self.emas[p][idx] for p in self.periods])
        prev_ema_vals = np.array([self.emas[p][idx-1] for p in self.periods])
        
        # Alignment check: each EMA > previous EMA in the sequence
        # And for trend: each current EMA > its previous value
        align_up = True
        for i in range(1, len(self.periods)):
            if current_ema_vals[i] <= current_ema_vals[i-1]:
                align_up = False
                break
        
        # Check if each EMA is actually increasing from last bar
        increasing = True
        for i in range(len(self.periods)):
            if current_ema_vals[i] <= prev_ema_vals[i]:
                increasing = False
                break

        # Compression check: std of EMAs / mean < threshold
        mean_ema = np.mean(current_ema_vals)
        std_ema = np.std(current_ema_vals)
        is_compressed = (std_ema / mean_ema) < self.threshold if mean_ema > 0 else False

        signal_type = "HOLD"
        confidence = 0.0
        spread = (np.max(current_ema_vals) - np.min(current_ema_vals))
        confidence = spread / current_close if current_close > 0 else 0.0

        if self.position is None:
            if align_up and increasing:
                self.position = "BUY"
                signal_type = "BUY"
        
        elif self.position == "BUY":
            if is_compressed:
                self.position = None
                signal_type = "SELL"
                confidence = 1.0

        return {
            "type": signal_type,
            "price": float(current_close),
            "confidence": float(confidence),
            "metadata": {
                "std_ema_mean_ratio": float(std_ema / mean_ema) if mean_ema > 0 else 0,
                "spread": float(spread)
            }
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"periods": self.periods, "compression_threshold": self.threshold}
