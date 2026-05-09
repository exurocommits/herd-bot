import numpy as np

class NFTFloorMomentum:
    """
    Simulate NFT floor price from volume data: 
    floor_price = base * (1 + 0.1 * volume_ma_ratio). 
    Floor momentum = floor ROC over 10 bars.
    
    BUY related token when floor momentum > 0 and accelerating.
    SELL when floor momentum negative.
    Confidence: floor_momentum / std(floor_momentum).
    """
    def __init__(self, data):
        """
        :param data: dict containing 'volume'
        """
        self.volumes = np.array(data['volume'])
        self.base_floor = 1000.0 # Simulated base floor
        self.lookback_vol = 20
        self.lookback_mom = 10
        self.floor_history = []

    def next(self, bar):
        volume = bar['volume']
        self.volumes = np.append(self.volumes, volume)

        if len(self.volumes) < self.lookback_vol:
            return {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {"reason": "warming_up"}}

        vol_ma = np.mean(self.volumes[-self.lookback_vol:])
        vol_ma_prev = np.mean(self.volumes[-self.lookback_vol-1:-1]) if len(self.volumes) > self.lookback_vol else vol_ma
        
        # volume_ma_ratio = current_ma / prev_ma (simulated as ratio of current vol to MA)
        vol_ma_ratio = volume / vol_ma if vol_ma > 0 else 1.0
        
        # floor_price = base * (1 + 0.1 * volume_ma_ratio)
        current_floor = self.base_floor * (1 + 0.1 * vol_ma_ratio)
        self.floor_history.append(current_floor)

        if len(self.floor_history) < self.lookback_mom + 1:
             return {"type": "HOLD", "price": float(current_floor), "confidence": 0.0, "metadata": {"reason": "warming_up"}}

        # Floor ROC over 10 bars
        floor_roc = (self.floor_history[-1] - self.floor_history[-self.lookback_mom-1]) / self.floor_history[-self.lookback_mom-1]
        
        # Acceleration: second derivative (change in ROC)
        # For simplicity, check if ROC is greater than previous ROC
        prev_floor_roc = (self.floor_history[-2] - self.floor_history[-self.lookback_mom-2]) / self.floor_history[-self.lookback_mom-2] if len(self.floor_history) > self.lookback_mom + 1 else 0
        
        signal_type = "HOLD"
        if floor_roc > 0 and floor_roc > prev_floor_roc:
            signal_type = "BUY"
        elif floor_roc < 0:
            signal_type = "SELL"

        # Confidence: floor_momentum / std(floor_momentum)
        # We need a window for std
        if len(self.floor_history) > 5:
            roc_history = []
            for i in range(1, len(self.floor_history)):
                roc_history.append((self.floor_history[i] - self.floor_history[i-1]) / self.floor_history[i-1])
            
            std_roc = np.std(roc_history[-10:]) if len(roc_history) > 2 else 0.01
            confidence = abs(floor_roc / std_roc) if std_roc > 0 else 0.0
        else:
            confidence = 0.0

        return {
            "type": signal_type,
            "price": float(current_floor),
            "confidence": float(min(confidence, 1.0)),
            "metadata": {
                "floor_roc": float(floor_roc),
                "floor_price": float(current_floor)
            }
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"base_floor": self.base_floor}
