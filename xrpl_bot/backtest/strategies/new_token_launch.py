import numpy as np

class NewTokenLaunch:
    """
    Strategy to detect and trade new token launches based on volume spikes.
    """

    def __init__(self, data):
        """
        Initialize the strategy.
        :param data: Historical data
        """
        self.data = data
        self.current_index = 0
        self.history = []
        self.in_trade = False
        self.entry_bar = 0
        self.volume_history = []
        self.price_history = []

    def next(self, bar):
        """
        Processes the next bar of data.
        :param bar: dict containing 'volume', 'price', 'bar_index'
        :return: dict containing signal information
        """
        self.current_index += 1
        volume = bar['volume']
        price = bar['price']
        bar_index = bar.get('bar_index', self.current_index)
        
        self.volume_history.append(volume)
        self.price_history.append(price)

        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}

        # Need some history for average volume
        if len(self.volume_history) < 20:
            self.history.append(signal)
            return signal

        avg_volume = np.mean(self.volume_history[:-1])
        
        # Potential launch: volume > 3x avg
        is_potential_launch = volume > (3 * avg_volume)

        # If not in trade, look for entry
        if not self.in_trade:
            if is_potential_launch:
                # Score launches
                liquidity = volume * price
                # age_penalty = 1 / (bar_since_launch + 1). 
                # In a real stream we'd track launch time. Here we assume launch is happening now.
                age_penalty = 1.0 
                
                # risk = 1 - liquidity_score. 
                # Let's define liquidity_score based on a normalized liquidity threshold.
                # For simulation, we'll use a simple log scale or threshold.
                liquidity_score = min(liquidity / 100000, 1.0) 
                risk_score = 1.0 - liquidity_score
                
                # Momentum positive: price > previous price
                momentum_positive = price > self.price_history[-2]

                if risk_score < 0.5 and momentum_positive:
                    self.in_trade = True
                    self.entry_bar = bar_index
                    confidence = 1.0 - risk_score
                    signal = {
                        "type": "BUY",
                        "price": float(price),
                        "confidence": float(confidence),
                        "metadata": {
                            "risk_score": float(risk_score),
                            "liquidity": float(liquidity)
                        }
                    }
        else:
            # In trade: SELL on first red bar after entry
            if price < self.price_history[-2]:
                self.in_trade = False
                signal = {
                    "type": "SELL",
                    "price": float(price),
                    "confidence": 1.0,
                    "metadata": {"reason": "red_bar"}
                }

        self.history.append(signal)
        return signal

    def get_signals(self):
        return self.history

    def get_params(self):
        return {}
