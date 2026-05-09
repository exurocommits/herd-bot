import numpy as np

class OnchainMLFeatures:
    """
    Simulate on-chain features and use a linear model for signals.
    Features: active_addr_change(10), hash_rate_change(10), price_momentum(5), volume_change(10), RSI.
    """
    def __init__(self, data):
        self.data = data
        self.noise_factor = 0.1
        self.current_index = 0

    def _get_features(self, idx):
        close = self.data['close'].values
        vol = self.data['volume'].values
        returns = np.diff(np.log(close), prepend=close[0])
        
        # Simulate on-chain features
        # active_addresses = volume * noise_factor
        active_addresses = vol * (1 + np.random.uniform(-self.noise_factor, self.noise_factor, len(vol)))
        # hash_rate_proxy = abs(return) * volume
        hash_rate_proxy = np.abs(returns) * vol
        
        # Feature calculation helper
        def get_change(arr, window):
            if idx < window: return 0.0
            return (arr[idx] - arr[idx-window]) / (arr[idx-window] + 1e-9)

        addr_change = get_change(active_addresses, 10)
        hash_change = get_change(hash_rate_proxy, 10)
        mom = (close[idx] / close[max(0, idx-5)] - 1) if idx >= 5 else 0
        vol_change = get_change(vol, 10)
        rsi = self.data['rsi'].iloc[idx]

        return np.array([addr_change, hash_change, mom, vol_change, rsi])

    def next(self, bar):
        idx = self.current_index
        price = self.data['close'].iloc[idx]
        feats = self._get_features(idx)
        
        addr_change, hash_change, momentum, vol_change, rsi = feats
        
        # Signal = 0.3*addr_change + 0.2*hash_change + 0.2*momentum + 0.15*vol_change + 0.15*(RSI-50)/50
        signal = (0.3 * addr_change + 
                  0.2 * hash_change + 
                  0.2 * momentum + 
                  0.15 * vol_change + 
                  0.15 * (rsi - 50) / 50)
        
        self.current_index += 1
        return {"price": price, "signal": signal}

    def get_signals(self):
        # This would normally be called in the loop. 
        # For the purpose of the class structure, we return the signal logic.
        return []

    def get_params(self):
        return {"noise_factor": self.noise_factor}

    def get_signal_dict(self, signal, price):
        if signal > 0.3:
            return {"type": "BUY", "price": float(price), "confidence": float(abs(signal)), "metadata": {}}
        elif signal < -0.3:
            return {"type": "SELL", "price": float(price), "confidence": float(abs(signal)), "metadata": {}}
        return {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
