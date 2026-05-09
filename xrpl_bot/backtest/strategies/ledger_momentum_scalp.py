import numpy as np

class LedgerMomentumScalp:
    \"\"\"
    XRPL tx count momentum strategy.
    Generates synthetic tx_count from volume.
    BUY when momentum (current tx_count / avg tx_count) > 1.5x avg.
    Window = 10.
    \"\"\"

    def __init__(self, data):
        \"\"\"
        :param data: pd.DataFrame or dict with 'close' and 'volume'
        \"\"\"
        self.data = data
        self.window = 10
        self.history = []
        self.current_index = 0
        
        # Pre-calculate tx_count if not provided
        # Synthetic: tx_count ~ volume * scaling_factor + noise
        if isinstance(data, dict):
            self.prices = np.array(data['close'])
            self.volumes = np.array(data['volume'])
        else:
            self.prices = data['close'].values
            self.volumes = data['volume'].values

        self.tx_counts = self.volumes * 0.5 + np.random.normal(0, self.volumes * 0.05, len(self.volumes))
        self.signals = []

    def next(self, bar):
        \"\"\"
        Processes a single bar.
        :param bar: dict containing 'close', 'volume', 'tx_count' (if provided)
        :return: dict signal
        \"\"\"
        idx = self.current_index
        price = bar['close']
        vol = bar['volume']
        
        # Use provided tx_count or synthetic
        tx_count = bar.get('tx_count', self.tx_counts[idx])
        
        self.history.append(tx_count)
        
        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
        
        if len(self.history) > self.window:
            avg_tx = np.mean(self.history[-(self.window + 1):-1])
            current_tx = self.history[-1]
            momentum = current_tx / avg_tx if avg_tx != 0 else 0
            
            if momentum > 1.5:
                signal = {
                    \"type\": \"BUY\",
                    \"price\": float(price),
                    \"confidence\": min(float(momentum / 3.0), 1.0),
                    \"metadata\": {\"momentum\": float(momentum), \"avg_tx\": float(avg_tx)}
                }
            elif momentum < 0.5:
                signal = {
                    \"type\": \"SELL\",
                    \"price\": float(price),
                    \"confidence\": min(float((1.0 - momentum) / 1.5), 1.0),
                    \"metadata\": {\"momentum\": float(momentum), \"avg_tx\": float(avg_tx)}
                }

        self.current_index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"window": self.window}
