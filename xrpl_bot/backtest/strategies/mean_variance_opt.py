import numpy as np

class MeanVarianceOpt:
    \"\"\"
    Mean-Variance Optimization Strategy.
    
    Tracks 5 assets' returns over 50 bars.
    Calculates mean returns and covariance matrix.
    Finds optimal weights using simplified Markowitz (max Sharpe).
    BUY the underweighted asset. SELL the overweighted.
    Rebalance when any weight deviates > 10% from target.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        :param data: dict containing 'prices' as a 2D numpy array (n_bars, n_assets).
        \"\"\"
        self.prices = data['prices']
        self.n_assets = self.prices.shape[1]
        self.window = 50
        self.current_index = 0
        self.target_weights = np.array([1.0/self.n_assets] * self.n_assets)
        self.last_weights = np.array([0.0] * self.n_assets)
        
    def _calculate_optimal_weights(self, returns):
        \"\"\"Simplified Markowitz: Maximize (w.T @ mu) / sqrt(w.T @ sigma @ w)\"\"\"
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns, rowvar=False)
        
        # To avoid complex quadratic programming, we'll use a simplified approach:
        # In a real production environment, we'd use scipy.optimize.
        # Here we'll use a simple heuristic for 'max Sharpe' in a self-contained numpy environment:
        # We'll try a small number of random weight vectors and pick the best, 
        # or just use the mean-return-to-volatility ratio as a proxy.
        
        # Let's try a Monte Carlo approach for the 'simplified' requirement
        best_sharpe = -np.inf
        best_w = np.array([1.0/self.n_assets] * self.n_assets)
        
        # Try 100 random weight combinations
        for _ in range(100):
            w = np.random.dirichlet(np.ones(self.n_assets), size=1)[0]
            # Portfolio return and vol
            p_ret = np.sum(mu * w)
            p_vol = np.sqrt(w.T @ sigma @ w)
            
            if p_vol > 0:
                sharpe = p_ret / p_vol
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = w
                    
        return best_w, best_sharpe

    def next(self, bar):
        \"\"\"
        :param bar: dict containing 'prices' (list/array of current prices for assets).
        \"\"\"
        self.current_index += 1
        
        # Update internal price history (assuming we are simulating)
        # In a real backtest, we'd append 'bar' to a history. 
        # For this implementation, we assume we have access to the full price array from __init__.
        
        idx = self.current_index
        if idx < self.window:
            return {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {}}
            
        # Calculate returns for the window
        window_prices = self.prices[idx-self.window : idx+1]
        returns = (window_prices[1:] - window_prices[:-1]) / window_prices[:-1]
        
        # Optimization
        target_weights, expected_sharpe = self._calculate_optimal_weights(returns)
        
        # Check rebalance condition
        # If we have no last_weights, we treat it as a rebalance
        rebalance_needed = False
        if np.all(self.last_weights == 0):
            rebalance_needed = True
        else:
            # Check if any weight deviates > 10% from target
            # Note: This is usually checked against current holdings, 
            # but we'll check deviation from target.
            diff = np.abs(target_weights - self.last_weights)
            if np.any(diff > 0.10):
                rebalance_needed = True
        
        signal = {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {}}
        
        if rebalance_needed:
            # Identify underweighted and overweighted
            # For simplicity, we'll just signal for the most extreme one
            # In a real bot, we'd execute multiple trades.
            for i in range(self.n_assets):
                if target_weights[i] > self.last_weights[i] + 0.10:
                    signal = {
                        "type": "BUY",
                        "price": float(self.prices[idx, i]),
                        "confidence": float(expected_sharpe),
                        "metadata": {"asset_idx": i, "target_w": float(target_weights[i])}
                    }
                    break
                elif target_weights[i] < self.last_weights[i] - 0.10:
                    signal = {
                        "type": "SELL",
                        "price": float(self.prices[idx, i]),
                        "confidence": float(expected_sharpe),
                        "metadata": {"asset_idx": i, "target_w": float(target_weights[i])}
                    }
                    break
            
            self.last_weights = target_weights.copy()

        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"window": self.window, "n_assets": self.n_assets}
