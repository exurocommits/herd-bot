import numpy as np

class SectorMomentumRotation:
    \"\"\"
    Sector Momentum Rotation Strategy.
    
    Takes 5 price series. Rank by 20-bar momentum (return).
    Long top 2, short bottom 2. Rebalance every 20 bars.
    BUY top-ranked, SELL bottom-ranked.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        :param data: dict containing 'prices' as a 2D numpy array (n_bars, n_assets).
        \"\"\"
        self.prices = data['prices']
        self.n_assets = self.prices.shape[1]
        self.momentum_window = 20
        self.rebalance_period = 20
        self.current_index = 0
        self.last_rebalance = 0
        
    def next(self, bar):
        \"\"\"
        :param bar: dict containing 'prices' (list/array of current prices).
        \"\"\"
        self.current_index += 1
        idx = self.current_index
        
        # Check if rebalance is due
        if idx < self.momentum_window or (idx - self.last_rebalance) < self.rebalance_period:
            return {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {}}
            
        # Calculate momentum (20-bar return)
        # momentum = (price_now - price_20_bars_ago) / price_20_bars_ago
        momentum = []
        for i in range(self.n_assets):
            p_now = self.prices[idx, i]
            p_prev = self.prices[idx - self.momentum_window, i]
            ret = (p_now - p_prev) / p_prev
            momentum.append(ret)
            
        momentum = np.array(momentum)
        # Rank assets (0 is lowest momentum, 4 is highest)
        ranks = np.argsort(momentum)
        
        # Top 2: ranks[-2:]
        # Bottom 2: ranks[:2]
        
        top_2 = ranks[-2:]
        bottom_2 = ranks[:2]
        
        avg_return = np.mean(momentum)
        
        # We'll signal the top-ranked asset (BUY) or bottom-ranked (SELL)
        # to satisfy the "BUY top-ranked, SELL bottom-ranked" instruction
        
        # Let's pick the best and worst for the signal
        best_idx = top_2[-1]
        worst_idx = bottom_2[0]
        
        # Signal logic:
        # This is a rotation strategy, so we return the primary action.
        # For the sake of a single 'next' return:
        # If we just rebalanced, we return the BUY signal for the top.
        # We'll use metadata to indicate both.
        
        confidence = (momentum[best_idx] - momentum[worst_idx]) / avg_return if avg_return != 0 else 0.0
        
        # In a real system, we would emit multiple signals.
        # Here we return the "BUY" signal for the top asset.
        signal = {
            "type": "BUY",
            "price": float(self.prices[idx, best_idx]),
            "confidence": float(abs(confidence)),
            "metadata": {
                "top_indices": top_2.tolist(),
                "bottom_indices": bottom_2.tolist(),
                "best_idx": int(best_idx),
                "worst_idx": int(worst_idx)
            }
        }
        
        self.last_rebalance = idx
        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"momentum_window": self.momentum_window, "rebalance_period": self.rebalance_period}
