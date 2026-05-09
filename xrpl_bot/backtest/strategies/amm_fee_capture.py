import numpy as np

class AMMFeeCapture:
    """
    LP strategy: monitor AMM pool spread (simulated from price volatility).
    Spread = 2 * (volatility * sqrt(time)).
    When spread > historical 80th percentile -> provide liquidity (BUY = add LP).
    When spread < 20th percentile -> remove liquidity (SELL = withdraw LP).
    Confidence: percentile_rank / 100.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close', 'volume'
        """
        self.prices = np.array(data['close'])
        self.volumes = np.array(data['volume'])
        self.spreads = []
        self.lookback = 100 # Window for percentile calculation

    def next(self, bar):
        price = bar['close']
        # Simulate volatility as standard deviation of last 10 returns
        # In a real app, this would be actual AMM spread data
        returns = np.diff(self.prices) if len(self.prices) > 1 else [0]
        volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0.01
        
        # time is assumed constant per bar (e.g., 1 minute)
        time_factor = 1.0 
        spread = 2 * (volatility * np.sqrt(time_factor))
        
        self.spreads.append(spread)
        
        if len(self.spreads) < self.lookback:
            return {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {"reason": "warming_up"}}

        # Calculate percentile rank
        recent_spreads = np.array(self.spreads[-self.lookback:])
        # percentile_rank logic: how many values are below current spread
        # Using scipy would be easier, but sticking to numpy/pure python as requested
        ranks = np.array([(spread > s).sum() for s in recent_spreads])
        percentile_rank = ranks[-1] / len(recent_spreads)

        signal_type = "HOLD"
        if percentile_rank > 0.8:
            signal_type = "BUY" # Add LP
        elif percentile_rank < 0.2:
            signal_type = "SELL" # Remove LP

        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(percentile_rank),
            "metadata": {"spread": float(spread), "percentile_rank": float(percentile_rank)}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"lookback": self.lookback}
