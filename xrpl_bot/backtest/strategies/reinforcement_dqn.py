import numpy as np

class ReinforcementDQN:
    """
    Simplified Deep Q-Network using a Tabular Q-learning approach.
    State: [return_1, return_5, return_10, RSI_norm, vol_ratio_norm]
    Actions: BUY=0, SELL=1, HOLD=2
    """
    def __init__(self, data):
        self.data = data
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.2  # 20% random
        self.bins = 10
        self.last_state = None
        self.last_action = None
        self.last_price = None
        self.current_index = 0

    def _get_discretized_state(self, state):
        # Discretize continuous features into bins
        return tuple(np.digitize(state, np.linspace(-1, 1, self.bins)) for _ in range(len(state)))

    def _get_state_features(self, idx):
        # Placeholder for feature extraction logic
        # In a real scenario, this would extract returns, RSI, etc.
        # For this implementation, we assume 'data' contains these or we derive them.
        # We'll simulate feature extraction based on the requirement.
        returns = self.data['close'].pct_change()
        rsi = self.data['rsi'] / 100.0 - 0.5 # norm
        vol_ratio = self.data['volume'] / self.data['volume'].rolling(20).mean()
        
        # Normalize features to [-1, 1] roughly for binning
        r1 = returns.iloc[idx] * 100
        r5 = returns.iloc[max(0, idx-5):idx+1].mean() * 100
        r10 = returns.iloc[max(0, idx-10):idx+1].mean() * 100
        
        return np.array([r1, r5, r10, rsi.iloc[idx], vol_ratio.iloc[idx]])

    def next(self, bar):
        idx = self.current_index
        price = self.data['close'].iloc[idx]
        state = self._get_state_features(idx)
        state_key = self._get_discretized_state(state)

        # Update Q-table with previous action's reward
        if self.last_state is not None and self.last_action is not None:
            reward = (price / self.last_price - 1) if self.last_action == 0 else \
                     (-price / self.last_price + 1) if self.last_action == 1 else 0
            
            old_q = self.q_table.get((self.last_state, self.last_action), 0.0)
            next_max_q = max([self.q_table.get((state_key, a), 0.0) for a in range(3)])
            new_q = old_q + self.learning_rate * (reward + self.discount * next_max_q - old_q)
            self.q_table[(self.last_state, self.last_action)] = new_q

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            q_values = [self.q_table.get((state_key, a), 0.0) for a in range(3)]
            action = np.argmax(q_values)

        self.last_state = state_key
        self.last_action = action
        self.last_price = price
        self.current_index += 1

        return {"price": price, "state": state_key, "action": action}

    def get_signals(self):
        # In a backtest loop, this is called after next()
        # For this implementation, we return the signal for the last processed bar
        # This is a simplified representation.
        return []

    def get_params(self):
        return {"learning_rate": self.learning_rate, "discount": self.discount, "epsilon": self.epsilon}
