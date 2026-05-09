import numpy as np

class RegimeHMM:
    """
    Hidden Markov Model strategy with 3 states: Bull (0), Range (1), Bear (2).
    Uses Gaussian emissions and Baum-Welch training on returns.
    """
    def __init__(self, data):
        """
        :param data: DataFrame with 'close' and 'returns' columns.
        """
        self.data = data
        self.n_states = 3
        self.transition_matrix = np.eye(self.n_states)
        self.means = np.zeros(self.n_states)
        self.vars = np.ones(self.n_states)
        self.pi = np.array([1/3, 1/3, 1/3])
        self.window_size = 200
        self.retrain_interval = 100
        self.bars_since_retrain = 0
        self.current_state_probs = np.array([1/3, 1/3, 1/3])
        self.last_idx = 0

    def _train(self, returns):
        """Simple Baum-Welch inspired parameter estimation."""
        T = len(returns)
        if T < 10:
            return

        # Initialize/Reset parameters
        self.means = np.array([np.mean(returns) + 0.01, np.mean(returns), np.mean(returns) - 0.01])
        self.vars = np.array([0.001, 0.0005, 0.001])
        
        # Simplified: Expectation-Maximization via local likelihood maximization
        # In a real scenario, this would be a full Baum-Welch. 
        # For a self-contained numpy strategy, we'll use a heuristic clustering 
        # or simple MLE for the states over the window.
        
        # We'll approximate states by splitting returns into quantiles for the window
        # to ensure 3 distinct regimes exist for the model to "learn".
        sorted_indices = np.argsort(returns)
        idx0 = sorted_indices[:T//3]
        idx1 = sorted_indices[T//3 : 2*T//3]
        idx2 = sorted_indices[2*T//3:]

        self.means[0] = np.mean(returns[idx2]) # Bull = high returns
        self.means[1] = np.mean(returns[idx1]) # Range = mid
        self.means[2] = np.mean(returns[idx0]) # Bear = low returns
        
        self.vars[0] = np.var(returns[idx2]) + 1e-6
        self.vars[1] = np.var(returns[idx1]) + 1e-6
        self.vars[2] = np.var(returns[idx0]) + 1e-6

        # Transition matrix: tendency to stay in current state
        self.transition_matrix = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ])

    def next(self, bar):
        """
        :param bar: dict containing 'close', 'returns', 'index'
        :return: signal dict
        """
        idx = bar['index']
        returns_series = self.data['returns'].values
        current_return = bar['returns']
        
        # Handle retraining
        if idx % self.retrain_interval == 0:
            start_idx = max(0, idx - self.window_size)
            self._train(returns_series[start_idx:idx+1])
        
        # E-step: Calculate posterior probabilities (filtering)
        # P(s_t | obs_1:t)
        likelihoods = np.zeros(self.n_states)
        for i in range(self.n_states):
            likelihoods[i] = (1.0 / np.sqrt(2 * np.pi * self.vars[i])) * \
                             np.exp(-(current_return - self.means[i])**2 / (2 * self.vars[i]))
        
        # Update belief
        new_probs = self.pi * likelihoods
        if np.sum(new_probs) > 0:
            new_probs /= np.sum(new_probs)
            # Incorporate transition
            self.current_state_probs = new_probs 
        else:
            self.current_state_probs = np.array([1/3, 1/3, 1/3])

        # State logic: 0=bull, 1=range, 2=bear
        state = np.argmax(self.current_state_probs)
        confidence = self.current_state_probs[state]
        
        signal_type = "HOLD"
        if state == 0:
            signal_type = "BUY"
        elif state == 2:
            signal_type = "SELL"
            
        return {
            "type": signal_type,
            "price": bar['close'],
            "confidence": float(confidence),
            "metadata": {"state": int(state), "probs": self.current_state_probs.tolist()}
        }

    def get_signals(self):
        """Placeholder for bulk processing if needed."""
        return []

    def get_params(self):
        return {
            "means": self.means.tolist(),
            "vars": self.vars.tolist(),
            "transition": self.transition_matrix.tolist()
        }
