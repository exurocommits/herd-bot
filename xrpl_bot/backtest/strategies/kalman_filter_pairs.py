import numpy as np

class KalmanFilterPairs:
    """
    Kalman Filter Pairs strategy.
    Uses a Kalman filter to estimate a dynamic hedge ratio beta and drift.
    Observation: price_a = beta * price_b + drift + noise.
    """
    def __init__(self, data):
        """
        :param data: dict with 'a' and 'b' as numpy arrays of prices.
        """
        self.price_a = data['a']
        self.price_b = data['b']
        self.window = 30 # for z-score calculation
        self.current_idx = 0
        self.position = 0 
        
        # Kalman state: [beta, drift]
        self.state = np.array([1.0, 0.0])
        # Covariance matrix
        self.P = np.eye(2) * 1.0
        # Process noise
        self.Q = np.array([[0.01, 0.0], [0.0, 0.01]])
        # Observation noise
        self.R = 1.0

    def next(self, bar_idx):
        """
        Processes the next bar using the Kalman filter update.
        :param bar_idx: Current index in the data series.
        :return: dict signal or None.
        """
        self.current_idx = bar_idx
        if bar_idx < 1:
            return None

        obs_a = self.price_a[bar_idx]
        obs_b = self.price_b[bar_idx]

        # --- Kalman Update ---
        # Prediction step
        # state_pred = state (constant model)
        # P_pred = P + Q
        state_pred = self.state
        P_pred = self.P + self.Q

        # Update step
        # H is the observation matrix: [price_b, 1]
        H = np.array([obs_b, 1.0])
        
        # residual y = obs_a - H * state_pred
        y = obs_a - np.dot(H, state_pred)
        
        # S = H * P_pred * H.T + R
        S = np.dot(H, np.dot(P_pred, H.T)) + self.R
        
        # K = P_pred * H.T / S
        K = np.dot(P_pred, H.T) / S
        
        # new state
        self.state = state_pred + K * y
        # new P
        self.P = P_pred - np.outer(K, np.dot(H, P_pred))

        # --- Signal Generation ---
        beta_est = self.state[0]
        drift_est = self.state[1]
        
        current_spread = obs_a - (beta_est * obs_b + drift_est)
        
        # To get Z-score, we need some history of spreads. 
        # Since we are online, we track it.
        if not hasattr(self, 'spread_history'):
            self.spread_history = []
        self.spread_history.append(current_spread)
        if len(self.spread_history) > self.window:
            self.spread_history.pop(0)

        if len(self.spread_history) < self.window:
            return None

        mean_spread = np.mean(self.spread_history)
        std_spread = np.std(self.spread_history)
        z_score = (current_spread - mean_spread) / std_spread if std_spread != 0 else 0

        signal = {"type": "HOLD", "price": obs_a, "confidence": 0.0, "metadata": {"z": z_score, "beta": beta_est}}

        if self.position == 0:
            if z_score < -2:
                self.position = 1
                signal = {"type": "BUY", "price": obs_a, "confidence": min(abs(z_score) / 3.0, 1.0), "metadata": {"z": z_score, "beta": beta_est}}
            elif z_score > 2:
                self.position = -1
                signal = {"type": "SELL", "price": obs_a, "confidence": min(abs(z_score) / 3.0, 1.0), "metadata": {"z": z_score, "beta": beta_est}}
        elif self.position == 1:
            if z_score >= 0:
                self.position = 0
                signal = {"type": "SELL", "price": obs_a, "confidence": 0.0, "metadata": {"z": z_score, "beta": beta_est}}
        elif self.position == -1:
            if z_score <= 0:
                self.position = 0
                signal = {"type": "BUY", "price": obs_a, "confidence": 0.0, "metadata": {"z": z_score, "beta": beta_est}}

        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"Q": 0.01, "R": 1.0}
