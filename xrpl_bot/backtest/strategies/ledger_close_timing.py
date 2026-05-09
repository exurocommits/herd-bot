import numpy as np

class LedgerCloseTiming:
    """
    XRPL ledgers close every 3-5 seconds. Simulate close timing patterns.
    before_ledger_close = price tends to move in direction of net flow.
    Track net flow = sum(volume * sign(close-open)) over last 3 bars.
    BUY when net_flow > 2*ATR before expected close.
    SELL on close reversal.
    Confidence: net_flow / (2*ATR).
    """
    def __init__(self, data):
        """
        :param data: dict containing 'open', 'close', 'high', 'low', 'volume'
        """
        self.opens = np.array(data['open'])
        self.closes = np.array(data['close'])
        self.highs = np.array(data['high'])
        self.lows = np.array(data['low'])
        self.volumes = np.array(data['volume'])
        
        self.net_flows = []
        self.atr_history = []
        self.window = 3

    def _calculate_atr(self, period=14):
        if len(self.closes) < period:
            return 0.0
        # Simplified ATR for simulation
        tr = np.maximum(self.highs[-period:] - self.lows[-period:], 
                        np.maximum(abs(self.highs[-period:] - self.closes[-period-1:-1]),
                                   abs(self.lows[-period:] - self.closes[-period-1:-1])))
        return np.mean(tr)

    def next(self, bar):
        price_close = bar['close']
        price_open = bar['open']
        price_high = bar['high']
        price_low = bar['low']
        volume = bar['volume']

        # Update state
        self.opens = np.append(self.opens, price_open)
        self.closes = np.append(self.closes, price_close)
        self.highs = np.append(self.highs, price_high)
        self.lows = np.append(self.lows, price_low)
        self.volumes = np.append(self.volumes, volume)

        # Calculate net flow: sum(volume * sign(close-open)) over last 3 bars
        if len(self.closes) >= self.window:
            net_flow = 0.0
            for i in range(len(self.closes) - self.window, len(self.closes)):
                diff = self.closes[i] - self.opens[i]
                net_flow += self.volumes[i] * np.sign(diff)
            self.net_flows.append(net_flow)
        else:
            return {"type": "HOLD", "price": float(price_close), "confidence": 0.0, "metadata": {"reason": "warming_up"}}

        # ATR calculation
        atr = self._calculate_atr()
        if atr <= 0:
             return {"type": "HOLD", "price": float(price_close), "confidence": 0.0, "metadata": {"reason": "warming_up"}}

        current_net_flow = self.net_flows[-1]
        
        # Logic: BUY when net_flow > 2*ATR. 
        # (In a real system we'd track close reversal separately)
        signal_type = "HOLD"
        if current_net_flow > 2 * atr:
            signal_type = "BUY"
        elif current_net_flow < -2 * atr:
            signal_type = "SELL"

        confidence = min(abs(current_net_flow / (2 * atr)), 1.0) if atr != 0 else 0.0

        return {
            "type": signal_type,
            "price": float(price_close),
            "confidence": float(confidence),
            "metadata": {
                "net_flow": float(current_net_flow),
                "atr": float(atr)
            }
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"window": self.window}
