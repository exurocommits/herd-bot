import numpy as np

class XRPVelocityStrategy:
    """
    Measures network activity via XRP velocity.
    
    Velocity = volume / (price * constant_supply_proxy).
    Rising velocity = more XRP changing hands.
    
    Logic:
    - BUY when velocity > 1.5x its 50-bar MA.
    - SELL when velocity < 0.67x MA.
    - Confidence: velocity / MA_velocity - 1.
    """
    def __init__(self, data):
        """
        data: dict containing 'volume', 'price'
        constant_supply_proxy: default 1e10 (arbitrary large constant)
        """
        self.volumes = np.array(data['volume'])
        self.prices = np.array(data['price'])
        self.supply_proxy = 1e10
        self.index = 0
        self.signals = []
        self.velocity_history = []
        self.in_position = False

    def next(self, bar):
        """
        bar: dict containing 'volume', 'price'
        """
        vol = bar['volume']
        price = bar['price']
        
        velocity = vol / (price * self.supply_proxy)
        self.velocity_history.append(velocity)
        
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}
        
        # Need at least 50 bars for MA
        if len(self.velocity_history) > 50:
            ma_velocity = np.mean(self.velocity_history[-51:-1]) # MA of previous 50
            
            if not self.in_position:
                if velocity > 1.5 * ma_velocity:
                    confidence = (velocity / ma_velocity) - 1
                    signal = {
                        "type": "BUY",
                        "price": price,
                        "confidence": float(confidence),
                        "metadata": {"velocity": velocity, "ma_velocity": ma_velocity}
                    }
                    self.in_position = True
            else:
                if velocity < 0.67 * ma_velocity:
                    signal = {
                        "type": "SELL",
                        "price": price,
                        "confidence": 1.0,
                        "metadata": {"reason": "velocity_drop"}
                    }
                    self.in_position = False

        self.signals.append(signal)
        self.index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"buy_threshold": 1.5, "sell_threshold": 0.67, "ma_period": 50}
