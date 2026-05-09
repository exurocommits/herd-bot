import numpy as np

class WeightedCloseReversion:
    """
    Weighted close = (2*close + high + low) / 4.
    Calculate its 20-period SMA and std.
    BUY when weighted close < SMA - 2*std.
    SELL when > SMA + 2*std.
    Exit at SMA cross.
    Confidence: distance from SMA / (2*std).
    """
    def __init__(self, data):
        self.close = np.array(data['close'], dtype=float)
        self.high = np.array(data['high'], dtype=float)
        self.low = np.array(data['low'], dtype=float)
        self.window = 20
        self.signals = []
        self.current_idx = len(self.close) - 1
        self.in_position = False
        self.position_type = None # "BUY" or "SELL"

    def _get_weighted_close(self, c, h, l):
        return (2 * c + h + l) / 4

    def next(self, bar):
        self.close = np.append(self.close, bar['close'])
        self.high = np.append(self.high, bar['high'])
        self.low = np.append(self.low, bar['low'])
        self.current_idx += 1
        
        if self.current_idx < self.window:
            return {"type": "HOLD", "price": float(bar['close']), "confidence": 0.0, "metadata": {}}

        # Calculate weighted closes for the window
        w_closes = self._get_weighted_close(
            self.close[-self.window:], 
            self.high[-self.window:], 
            self.low[-self.window:]
        )
        
        sma = np.mean(w_closes)
        std = np.std(w_closes)
        current_w_close = w_closes[-1]
        price = float(bar['close'])
        
        signal_type = "HOLD"
        conf = 0.0
        
        if std > 0:
            diff = current_w_close - sma
            dist_factor = abs(diff) / (2 * std)
            conf = float(np.clip(dist_factor, 0, 1))

            # Logic
            if not self.in_position:
                if current_w_close < (sma - 2 * std):
                    signal_type = "BUY"
                    self.in_position = True
                    self.position_type = "BUY"
                elif current_w_close > (sma + 2 * std):
                    signal_type = "SELL"
                    self.in_position = True
                    self.position_type = "SELL"
            else:
                # Exit at SMA cross
                # If BUY, exit if current_w_close crosses SMA (from below or above)
                # Prompt says "Exit at SMA cross". 
                # Usually, if we are LONG, we exit if price <= SMA or >= SMA? 
                # Let's assume exit is when current_w_close crosses the SMA.
                prev_w_close = w_closes[-2]
                crossed = (prev_w_close < sma and current_w_close >= sma) or \
                          (prev_w_close > sma and current_w_close <= sma)
                
                if crossed:
                    signal_type = "EXIT" # Or just HOLD if we want to close. 
                    # The prompt implies we return signals. Let's use "HOLD" or "SELL" for exit.
                    # We'll use "SELL" if we were long, "BUY" (to cover) if short.
                    # But since the signal dict only has BUY/SELL/HOLD:
                    # Let's return HOLD and reset position.
                    signal_type = "HOLD" 
                    self.in_position = False
                    self.position_type = None

        signal = {"type": signal_type, "price": price, "confidence": conf, "metadata": {"sma": float(sma), "std": float(std)}}
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"window": self.window}
