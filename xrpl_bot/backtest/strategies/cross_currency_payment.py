import numpy as np

class CrossCurrencyPaymentStrategy:
    """
    Simulates large cross-currency payments and front-runs momentum.
    
    Logic:
    - Generates/detects large payment events (size > 5x avg volume).
    - When a large buy payment is detected (volume spike + price uptick) -> BUY.
    - Sell after 5 bars.
    - Confidence: payment_size / avg_volume.
    """
    def __init__(self, data):
        """
        data: dict containing 'volume' (list or numpy array) and 'price' (list or numpy array)
        """
        self.volumes = np.array(data['volume'])
        self.prices = np.array(data['price'])
        self.index = 0
        self.signals = []
        self.hold_timer = 0
        self.avg_volume = np.mean(self.volumes) if len(self.volumes) > 0 else 1.0

    def next(self, bar):
        """
        bar: dict containing 'volume' and 'price'
        """
        current_vol = bar['volume']
        current_price = bar['price']
        
        # Use previous price to detect uptick
        prev_price = self.prices[self.index - 1] if self.index > 0 else current_price
        price_uptick = current_price > prev_price
        
        signal = {"type": "HOLD", "price": current_price, "confidence": 0.0, "metadata": {}}
        
        # If we are in a trade, wait 5 bars to sell
        if self.hold_timer > 0:
            self.hold_timer -= 1
            if self.hold_timer == 0:
                signal = {
                    "type": "SELL", 
                    "price": current_price, 
                    "confidence": 1.0, 
                    "metadata": {"reason": "exit_after_5_bars"}
                }
            self.signals.append(signal)
            self.index += 1
            return signal

        # Detect large buy payment
        if current_vol > 5 * self.avg_volume and price_uptick:
            confidence = float(current_vol / self.avg_volume)
            self.hold_timer = 5
            signal = {
                "type": "BUY",
                "price": current_price,
                "confidence": confidence,
                "metadata": {"reason": "large_payment_momentum", "volume_spike": current_vol}
            }
        
        self.signals.append(signal)
        self.index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"spike_threshold": 5.0, "hold_period": 5}
