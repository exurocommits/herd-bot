import numpy as np

class StablecoinPegDevStrategy:
    """
    Tracks stablecoin price and trades based on peg deviations.
    
    Logic:
    - Deviation = price - 1.0
    - BUY when price < 0.995 (below peg).
    - SELL when price > 1.005 (above peg).
    - Exit at 1.0.
    - Confidence: |deviation| / 0.005.
    """
    def __init__(self, data):
        """
        data: dict containing 'price' (list or numpy array)
        """
        self.prices = np.array(data['price'])
        self.index = 0
        self.signals = []
        self.in_position = False

    def next(self, bar):
        """
        bar: dict containing 'price'
        """
        price = bar['price']
        deviation = price - 1.0
        
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}
        
        if not self.in_position:
            if price < 0.995:
                confidence = abs(deviation) / 0.005
                signal = {
                    "type": "BUY",
                    "price": price,
                    "confidence": float(confidence),
                    "metadata": {"reason": "below_peg"}
                }
                self.in_position = True
        else:
            # Exit conditions
            # 1. Price goes above peg
            if price > 1.005:
                signal = {
                    "type": "SELL",
                    "price": price,
                    "confidence": 1.0,
                    "metadata": {"reason": "above_peg"}
                }
                self.in_position = False
            # 2. Price returns to peg
            elif abs(deviation) < 1e-5:
                signal = {
                    "type": "SELL",
                    "price": price,
                    "confidence": 1.0,
                    "metadata": {"reason": "returned_to_peg"}
                }
                self.in_position = False

        self.signals.append(signal)
        self.index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"lower_peg": 0.995, "upper_peg": 1.005}
