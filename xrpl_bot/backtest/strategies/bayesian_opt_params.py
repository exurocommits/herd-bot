import numpy as np

class BayesianOptEMA:
    """
    EMA Crossover strategy with simplified Bayesian Optimization (Thompson Sampling).
    """
    def __init__(self, data):
        """
        :param data: DataFrame with 'close' and 'returns' columns.
        """
        self.data = data
        self.fast_ema = 9
        self.slow_ema = 21
        self.best_params = (9, 21)
        self.best_sharpe = -np.inf
        
        # Thompson Sampling state
        # We track successes (Sharpe > 0) for parameter combinations
        # success_counts[params] = [alpha, beta]
        self.param_stats = {} 
        self.window_size = 50
        self.retrain_interval = 50
        self.last_idx = 0
        
        # To calculate rolling Sharpe
        self.returns_history = []

    def _get_ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def _evaluate_params(self, fast, slow, end_idx):
        """Evaluates a parameter set over the last window_size bars."""
        start_idx = max(0, end_idx - self.window_size)
        subset = self.data.iloc[start_idx:end_idx+1].copy()
        if len(subset) < 20:
            return -1.0, 0.0 # Sharpe, success_rate
            
        fast_ema = subset['close'].ewm(span=fast, adjust=False).mean()
        slow_ema = subset['close'].ewm(span=slow, adjust=False).mean()
        
        # Signal: 1 if fast > slow else -1
        signals = (fast_ema > slow_ema).astype(int).shift(1)
        strategy_returns = signals * subset['returns']
        
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9) * np.sqrt(252)
        success = 1.0 if sharpe > 0 else 0.0
        return sharpe, success

    def _optimize(self, current_idx):
        """Try 5 parameter combos using Thompson Sampling."""
        # Possible range for fast/slow
        # We'll sample from a discrete set for simplicity
        possible_fast = [5, 7, 9, 12, 15, 20]
        possible_slow = [21, 30, 40, 50, 60]
        
        # Ensure fast < slow
        combos = []
        for f in possible_fast:
            for s in possible_slow:
                if f < s:
                    combos.append((f, s))
        
        # Thompson Sampling: sample from Beta(alpha, beta)
        sampled_combos = []
        for combo in combos:
            stats = self.param_stats.get(combo, [1, 1]) # Default prior Beta(1,1)
            # Sample probability of success from Beta distribution
            prob_success = np.random.beta(stats[0], stats[1])
            sampled_combos.append((combo, prob_success))
        
        # Sort by sampled probability and pick top 5
        sampled_combos.sort(key=lambda x: x[1], reverse=True)
        to_try = [c[0] for c in sampled_combos[:5]]
        
        for f, s in to_try:
            sharpe, success = self._evaluate_params(f, s, current_idx)
            
            # Update stats
            stats = self.param_stats.get((f, s), [1, 1])
            new_stats = [stats[0] + success, stats[1] + (1 - success)]
            self.param_stats[(f, s)] = new_stats
            
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_params = (f, s)

    def next(self, bar):
        idx = bar['index']
        self.last_idx = idx
        
        if idx % self.retrain_interval == 0:
            self._optimize(idx)
            
        fast, slow = self.best_params
        
        # Compute EMAs for the current bar
        # In a real streaming context, we'd maintain the EMA state.
        # For simplicity here, we recalculate using the window.
        start_idx = max(0, idx - 100)
        subset = self.data.iloc[start_idx:idx+1].copy()
        
        fast_ema_series = subset['close'].ewm(span=fast, adjust=False).mean()
        slow_ema_series = subset['close'].ewm(span=slow, adjust=False).mean()
        
        f_val = fast_ema_series.iloc[-1]
        s_val = slow_ema_series.iloc[-1]
        
        sig_type = "HOLD"
        if f_val > s_val:
            sig_type = "BUY"
        elif f_val < s_val:
            sig_type = "SELL"
            
        # Confidence calculation
        # We'll use a normalized Sharpe-based confidence
        max_possible_sharpe = 3.0 # Heuristic
        conf = max(0.0, min(1.0, self.best_sharpe / max_possible_sharpe))

        return {
            "type": sig_type,
            "price": bar['close'],
            "confidence": float(conf),
            "metadata": {"fast": fast, "slow": slow, "best_sharpe": self.best_sharpe}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"fast": self.best_params[0], "slow": self.best_params[1]}
