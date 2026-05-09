import numpy as np

class DexCexPriceGap:
    """
    Strategy that tracks the gap between DEX mid-price and CEX price.
    """

    def __init__(self, data):
        """
        Initialize the strategy.
        :param data: Historical data
        """
        self.data = data
        self.min_gap = 0.003
        self.current_index = 0
        self.history = []

    def next(self, bar):
        """
        Processes the next bar of data.
        :param bar: dict containing 'dex_price' and 'cex_price'
        :return: dict containing signal information
        """
        self.current_index += 1
        
        dex_price = bar['dex_price']
        cex_price = bar['cex_price']

        # Gap = (dex_price - cex_price) / cex_price
        gap = (dex_price - cex_price) / cex_price
        
        signal = {"type": "HOLD", "price": float(dex_price), "confidence": 0.0, "metadata": {}}

        # BUY on DEX when gap < -0.003 (DEX cheap)
        if gap < -self.min_gap:
            # Confidence: |gap| / 0.003
            confidence = min(abs(gap) / self.min_gap, 1.0)
            signal = {
                "type": "BUY",
                "price": float(dex_price),
                "confidence": float(confidence),
                "metadata": {
                    "gap": float(gap),
                    "cex_price": float(cex_price)
                }
            }

        # SELL on DEX when gap > 0.003
        elif gap > self.min_gap:
            confidence = min(abs(gap) / self.min_gap, 1.0)
            signal = {
                "type": "SELL",
                "price": float(dex_price),
                "confidence": float(confidence),
                "metadata": {
                    "gap": float(gap),
                    "cex_price": float(cex_price)
                }
            }

        self.history.append(signal)
        return signal

    def get_signals(self):
        return self.history

    def get_params(self):
        return {"min_gap": self.min_gap}
