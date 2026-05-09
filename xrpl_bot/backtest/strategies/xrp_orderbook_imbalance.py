import numpy as np

class XRPOrderbookImbalance:
    \"\"\"
    XRP Orderbook Imbalance strategy.
    Uses bid/ask depth estimation from OHLCV.
    imbalance = bid_vol / (bid_vol + ask_vol).
    BUY when imbalance > 0.6.
    \"\"\"

    def __init__(self, data):
        \"\"\"
        :param data: pd.DataFrame or dict with 'close', 'high', 'low', 'volume'
        \"\"\"
        self.data = data
        self.history = []
        self.current_index = 0
        
        if isinstance(data, dict):
            self.prices = np.array(data['close'])
            self.highs = np.array(data['high'])
            self.lows = np.array(data['low'])
            self.volumes = np.array(data['volume'])
        else:
            self.prices = data['close'].values
            self.highs = data['high'].values
            self.lows = data['low'].values
            self.volumes = data['volume'].values

    def _estimate_imbalance(self, bar):
        \"\"\"
        Estimates imbalance using OHLCV as a proxy for order flow.
        A simple proxy: higher close relative to range implies bid pressure.
        \"\"\"
        high = bar['high']
        low = bar['low']
        close = bar['close']
        vol = bar['volume']
        
        range_val = high - low
        if range_val == 0:
            return 0.5
            
        # Relative position of close in range [0, 1]
        # 1.0 means close = high (bullish/bid pressure)
        # 0.0 means close = low (bearish/ask pressure)
        relative_pos = (close - low) / range_val
        
        # Map relative_pos to imbalance (0.5 is neutral)
        # If pos is 1.0, imbalance should be high (e.g., 0.8)
        # If pos is 0.0, imbalance should be low (e.g., 0.2)
        imbalance = 0.2 + (relative_pos * 0.6)
        return imbalance

    def next(self, bar):
        idx = self.current_index
        price = bar['close']
        
        imbalance = self._estimate_imbalance(bar)
        self.history.append(imbalance)
        
        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
        
        if imbalance > 0.6:
            signal = {
                "type": "BUY",
                "price": float(price),
                "confidence": float(min((imbalance - 0.6) / 0.4, 1.0)),
                "metadata": {"imbalance": float(imbalance)}
            }
        elif imbalance < 0.4:
            signal = {
                "type": "SELL",
                "price": float(price),
                "confidence": float(min((0.4 - imbalance) / 0.4, 1.0)),
                "metadata": {"imbalance": float(imbalance)}
            }

        self.current_index += 1
        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"imbalance_threshold_buy": 0.6, "imbalance_threshold_sell": 0.4}
