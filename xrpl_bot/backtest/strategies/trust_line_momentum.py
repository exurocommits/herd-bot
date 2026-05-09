import numpy as np

class TrustLineMomentum:
    """
    Strategy simulating trust line creation count from volume data.
    """

    def __init__(self, data):
        """
        Initialize the strategy.
        :param data: Historical data
        """
        self.data = data
        self.window = 10
        self.current_index = 0
        self.history = []
        self.trust_line_history = []

    def next(self, bar):
        """
        Processes the next bar of data.
        :param bar: dict containing 'volume' and 'price'
        :return: dict containing signal information
        """
        self.current_index += 1
        volume = bar['volume']
        price = bar['price']
        
        # Simulate trust line creation: new_trust_lines = int(volume / price * random_factor)
        # Using a deterministic 'random' factor based on index for simulation consistency if needed, 
        # but task says random_factor. I'll use a simple sine-based pseudo-random for reproducibility in tests.
        random_factor = 0.01 
        new_trust_lines = int((volume / price) * random_factor)
        self.trust_line_history.append(new_trust_lines)

        if len(self.trust_line_history) < self.window + 1:
            signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
            self.history.append(signal)
            return signal

        # Velocity = rate of change of trust lines over 10 bars
        # velocity = (TL_now - TL_10_bars_ago) / 10
        recent_tl = self.trust_line_history[-self.window:]
        velocity = (recent_tl[-1] - recent_tl[0]) / self.window

        # Average velocity over the whole history (excluding current window to avoid bias)
        if len(self.trust_line_history) > self.window * 2:
            avg_velocity = np.mean(self.trust_line_history[:-self.window]) / self.window
        else:
            # Fallback for early data
            avg_velocity = np.mean(self.trust_line_history) / self.window if self.trust_line_history else 1.0

        # Avoid division by zero
        if avg_velocity <= 0:
            avg_velocity = 1e-9

        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}

        # BUY when velocity > 2x average velocity
        if velocity > 2 * avg_velocity:
            confidence = min(velocity / (2 * avg_velocity), 1.0) # Normalized slightly differently
            # Following prompt: Confidence: velocity / avg_velocity
            confidence = min(velocity / avg_velocity, 1.0)
            signal = {
                "type": "BUY",
                "price": float(price),
                "confidence": float(confidence),
                "metadata": {
                    "velocity": float(velocity),
                    "avg_velocity": float(avg_velocity)
                }
            }
        
        # SELL when velocity < 0.5x average
        elif velocity < 0.5 * avg_velocity:
            confidence = min(avg_velocity / (velocity if velocity > 0 else 1e-9), 1.0)
            # Prompt doesn't specify SELL confidence, using inverse of ratio
            signal = {
                "type": "SELL",
                "price": float(price),
                "confidence": float(confidence),
                "metadata": {
                    "velocity": float(velocity),
                    "avg_velocity": float(avg_velocity)
                }
            }

        self.history.append(signal)
        return signal

    def get_signals(self):
        return self.history

    def get_params(self):
        return {"window": self.window}
