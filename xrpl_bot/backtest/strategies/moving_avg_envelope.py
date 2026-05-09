import numpy as np

class MovingAvgEnvelope:
    """
    20 SMA ± 3% envelope.
    BUY when price touches lower envelope band (close < SMA * 0.97).
    SELL when touches upper (close > SMA * 1.03).
    Exit at SMA.
    Confidence: distance from SMA / envelope width.
    """
    def __init__(self, data):
        """
        :param data: Dictionary containing 'close' as numpy array.
        """
        self.close = data['close']
        self.period = 20
        self.current_index = 0
        self.position = None  # None, 'BUY', 'SELL'
        self.sma_history = None

    def _calculate_sma(self, idx):
        if idx < self.period - 1:
            return None
        return np.mean(self.close[idx - self.period + 1 : idx + 1])

    def next(self, bar):
        idx = self.current_index
        self.current_index += 1
        
        current_close = bar['close']
        sma = self._calculate_sma(idx)
        
        if sma is None:
            return {"type": "HOLD", "price": current_close, "confidence": 0.0, "metadata": {}}

        upper_band = sma * 1.03
        lower_band = sma * 0.97
        envelope_width = upper_band - lower_band
        
        signal_type = "HOLD"
        confidence = 0.0
        metadata = {
            "sma": float(sma),
            "upper_band": float(upper_band),
            "lower_band": float(lower_band),
            "envelope_width": float(envelope_width)
        }

        if self.position is None:
            if current_close < lower_band:
                self.position = "BUY"
                signal_type = "BUY"
                dist = sma - current_close
                confidence = dist / envelope_width if envelope_width > 0 else 0.0
            elif current_close > upper_band:
                self.position = "SELL"
                signal_type = "SELL"
                dist = current_close - sma
                confidence = dist / envelope_width if envelope_width > 0 else 0.0
        
        elif self.position == "BUY":
            # Exit at SMA
            # If price crosses SMA from below
            if current_close >= sma:
                self.position = None
                signal_type = "SELL"
                confidence = 1.0
            elif current_close > upper_band:
                # Already in BUY, price went even higher? Prompt says SELL on upper.
                # For a simple bot, let's treat upper touch as a signal to SELL (exit/profit).
                self.position = None
                signal_type = "SELL"
                confidence = 1.0
        
        elif self.position == "SELL":
            if current_close <= sma:
                self.position = None
                signal_type = "BUY"
                confidence = 1.0
            elif current_close < lower_band:
                self.position = None
                signal_type = "BUY"
                confidence = 1.0

        return {"type": signal_type, "price": current_close, "confidence": float(confidence), "metadata": metadata}

    def get_signals(self):
        return []

    def get_params(self):
        return {"period": self.period, "upper_mult": 1.03, "lower_mult": 0.97}
