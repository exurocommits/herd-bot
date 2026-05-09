import numpy as np

class LeadLagEffect:
    \"\"\"
    Lead-Lag Effect Strategy.
    
    Takes two price series. Finds optimal lag (1-10) where cross-correlation peaks.
    If Asset A leads Asset B by N bars:
    - BUY B when A had positive return N bars ago.
    - SELL B when A had negative return N bars ago.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        :param data: dict containing 'asset_a_price' and 'asset_b_price' as numpy arrays.
        \"\"\"
        self.a_prices = data['asset_a_price']
        self.b_prices = data['asset_b_price']
        self.max_lag = 10
        self.current_index = 0
        self.returns_a = np.diff(self.a_prices, prepend=self.a_prices[0])
        self.returns_b = np.diff(self.b_prices, prepend=self.b_prices[0])

    def next(self, bar):
        \"\"\"
        :param bar: dict containing 'asset_a_price' and 'asset_b_price' for current bar.
        \"\"\"
        self.current_index += 1
        # In a real backtest, we'd update returns based on the current bar.
        # For simplicity in this class structure, we assume data passed in __init__ is the full series,
        # and next() is called sequentially.
        
        # However, the prompt implies we use the historical window to find the lag.
        # Let's assume we have access to the historical window up to current_index.
        # We'll use the full series provided in __init__ but only look up to self.current_index.
        
        idx = self.current_index
        if idx < self.max_lag + 1:
            return {"type": "HOLD", "price": bar['asset_b_price'], "confidence": 0.0, "metadata": {}}

        # Get returns up to current index
        ret_a = self.returns_a[:idx+1]
        ret_b = self.returns_b[:idx+1]

        # Find optimal lag (1 to 10)
        best_lag = 1
        max_corr = -1.0
        
        for lag in range(1, self.max_lag + 1):
            if len(ret_a) > lag + 1 and len(ret_b) > lag + 1:
                # Correlation between ret_a[t-lag] and ret_b[t]
                # We align ret_a[:-lag] with ret_b[lag:]? No, A leads B.
                # If A leads B by N, then A's return at t-N predicts B's return at t.
                # Correlation(A[t-N], B[t])
                c = np.corrcoef(ret_a[:len(ret_b)-lag], ret_b[lag:])[0, 1]
                if not np.isnan(c) and c > max_corr:
                    max_corr = c
                    best_lag = lag

        # Check signal: if A's return at (idx - best_lag) was positive/negative
        # We need the return at index - best_lag
        prev_a_return = self.returns_a[idx - best_lag]
        
        signal_type = "HOLD"
        if prev_a_return > 0:
            signal_type = "BUY"
        elif prev_a_return < 0:
            signal_type = "SELL"
            
        return {
            "type": signal_type,
            "price": float(bar['asset_b_price']),
            "confidence": float(max_corr),
            "metadata": {"lag": best_lag}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"max_lag": self.max_lag}
