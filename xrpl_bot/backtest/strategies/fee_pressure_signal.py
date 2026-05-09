import numpy as np

class FeePressureSignal:
    """
    Simulates XRPL transaction fees from volume data to identify network demand spikes.
    
    Fee pressure = current_fee / 50-bar MA of fees.
    BUY XRP when fee_pressure > 1.5 (network demand spike).
    SELL when fee_pressure < 0.7 (demand dropping).
    """
    def __init__(self, data):
        """
        Initialize with data.
        :param data: dict containing 'close', 'volume', 'base_fee' (float)
        """
        self.data = data
        self.prices = np.array(data['close'])
        self.volumes = np.array(data['volume'])
        self.base_fee = data.get('base_fee', 0.00001)
        self.history = [] # List of (price, volume, fee)
        
    def next(self, bar):
        """
        Process the next bar.
        :param bar: dict with 'close', 'volume'
        :return: dict signal
        """
        price = bar['close']
        volume = bar['volume']
        
        # Simulate fee: fee_avg = base_fee * (1 + volume_spike_factor)
        # volume_spike_factor is simulated here as volume relative to a recent mean
        # In a real scenario, this would be actual fee data.
        # For simulation: we use volume as a proxy for demand.
        vol_ma = np.mean(self.volumes[-20:]) if len(self.volumes) > 20 else np.mean(self.volumes)
        volume_spike_factor = (volume / vol_ma - 1) if vol_ma > 0 else 0
        current_fee = self.base_fee * (1 + max(0, volume_spike_factor))
        
        self.history.append(current_fee)
        
        # We need at least 50 bars for the MA
        if len(self.history) < 50:
            return {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {"reason": "warming_up"}}
            
        fee_ma = np.mean(self.history[-50:])
        fee_pressure = current_fee / fee_ma if fee_ma > 0 else 1.0
        
        signal_type = "HOLD"
        if fee_pressure > 1.5:
            signal_type = "BUY"
        elif fee_pressure < 0.7:
            signal_type = "SELL"
            
        confidence = min(fee_pressure / 2.0, 1.0)
        
        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {"fee_pressure": float(fee_pressure), "current_fee": float(current_fee)}
        }

    def get_signals(self):
        """Placeholder for vectorized signal generation if needed."""
        return []

    def get_params(self):
        return {"base_fee": self.base_fee}
