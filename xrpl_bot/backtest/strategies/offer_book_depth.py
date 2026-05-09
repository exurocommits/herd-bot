import numpy as np

class OfferBookDepthStrategy:
    """
    Simulates order book depth based on price ranges.
    
    Logic:
    - bid_depth = estimated from low-close range.
    - ask_depth = estimated from close-high range.
    - Imbalance = bid_depth / (bid_depth + ask_depth).
    - BUY when imbalance > 0.65 (buying pressure).
    - SELL when imbalance < 0.35 (selling pressure).
    - Exit at 0.5.
    - Confidence: |imbalance - 0.5| / 0.2.
    """
    def __init__(self, data):
        """
        data: dict containing 'low', 'high', 'close'
        """
        self.prices = np.array(data['close'])
        self.index = 0
        self.signals = []
        self.in_position = False

    def next(self, bar):
        """
        bar: dict containing 'low', 'high', 'close'
        """
        low = bar['low']
        high = bar['high']
        close = bar['close']
        
        # Estimate depth from ranges relative to close
        # bid_depth is driven by how much the low is below the close
        # ask_depth is driven by how much the high is above the close
        bid_depth = close - low + 1e-9
        ask_depth = high - close + 1e-9
        
        imbalance = bid_depth / (bid_depth + ask_depth)
        
        signal = {"type": "HOLD", "price": close, "confidence": 0.0, "metadata": {}}
        
        if not self.in_position:
            if imbalance > 0.65:
                confidence = abs(imbalance - 0.5) / 0.2
                signal = {
                    "type": "BUY",
                    "price": close,
                    "confidence": float(confidence),
                    "metadata": {"imbalance": imbalance}
                }
                self.in_position = True
        else:
            if imbalance < 0.35:
                signal = {
                    "type": "SELL",
                    "price": close,
                    "confidence": 1.0,
                    "metadata": {"reason": "selling_pressure"}
                }
                self.in_position = False
            elif abs(imbalance - 0.5) < 1e-3:
                signal = {
                    "type": "SELL",
                    "price": close,
                    "confidence": 1.0,
                    "metadata": {"reason": "exit_at_neutral"}
                }
                self.in_position = False

        self.signals.append(signal)
        self.index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"buy_imbalance": 0.65, "sell_imbalance": 0.35, "exit_imbalance": 0.5}
