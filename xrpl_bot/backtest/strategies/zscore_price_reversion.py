import numpy as np

class ZScorePriceReversion:
    """
    Z-score of log returns over 50 bars.
    BUY when z < -2 (extreme negative).
    SELL when z > +2.
    Exit at z = 0.
    Confidence: |z| / 3.
    """
    def __init__(self, data):
        self.close = np.array(data['close'], dtype=float)
        self.window = 50
        self.signals = []
        self.current_idx = len(self.close) - 1
        self.in_position = False
        self.position_type = None # "BUY" or "SELL"

    def next(self, bar):
        self.close = np.append(self.close, bar['close'])
        self.current_idx += 1

        if self.current_idx < self.window + 1:
            return {"type": "HOLD", "price": float(bar['close']), "confidence": 0.0, "metadata": {}}

        # Log returns
        log_returns = np.diff(np.log(self.close[-(self.window+1):]))
        
        # Z-score of the latest log return
        mean_ret = np.mean(log_returns)
        std_ret = np.std(log_returns)
        
        latest_ret = log_returns[-1]
        
        if std_ret == 0:
            z = 0.0
        else:
            z = (latest_ret - mean_ret) / std_ret
            
        price = float(bar['close'])
        signal_type = "HOLD"
        conf = float(np.clip(abs(z) / 3.0, 0, 1))

        if not self.in_position:
            if z < -2:
                signal_type = "BUY"
                self.in_position = True
                self.position_type = "BUY"
            elif z > 2:
                signal_type = "SELL"
                self.in_position = True
                self.position_type = "SELL"
        else:
            # Exit at z = 0
            if (self.position_type == "BUY" and z >= 0) or \
               (self.position_type == "SELL" and z <= 0):
                signal_type = "HOLD" # We'll assume "HOLD" handles the exit logic in the caller 
                                     # or we just stop emitting signals.
                                     # To be safe, let's return a signal that signals an exit.
                                     # The prompt says "Exit at z = 0".
                                     # I'll return "SELL" if long, "BUY" if short to represent closing.
                if self.position_type == "BUY":
                    signal_type = "SELL"
                else:
                    signal_type = "BUY"
                
                self.in_position = False
                self.position_type = None

        signal = {"type": signal_type, "price": price, "confidence": conf, "metadata": {"z": float(z)}}
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"window": self.window}
