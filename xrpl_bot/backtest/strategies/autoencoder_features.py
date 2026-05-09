import numpy as np

class AutoencoderFeatures:
    """
    Uses a simple numpy-based online autoencoder to extract compressed features.
    Input: [return, vol_change, RSI_norm, ATR_norm, BB%B]
    Bottleneck: 3 dimensions.
    """
    def __init__(self, data):
        self.data = data
        self.input_dim = 5
        self.hidden_dim = 3
        self.lr = 0.01
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.W2 = np.random.randn(self.hidden_dim, self.input_dim) * 0.1
        
        self.current_index = 0

    def _get_input(self, idx):
        if idx == 0:
            return np.zeros(self.input_dim)
        
        ret = self.data['close'].pct_change().iloc[idx]
        vol_change = self.data['volume'].pct_change().iloc[idx]
        rsi_norm = (self.data['rsi'].iloc[idx] - 50) / 50
        atr_norm = self.data['atr'].iloc[idx] / self.data['close'].iloc[idx] if 'atr' in self.data.columns else 0.0
        bb_p_b = self.data['bb_p_b'].iloc[idx] if 'bb_p_b' in self.data.columns else 0.5
        
        return np.array([ret, vol_change, rsi_norm, atr_norm, bb_p_b])

    def next(self, bar):
        idx = self.current_index
        price = self.data['close'].iloc[idx]
        x = self._get_input(idx)
        
        # Forward pass
        z = np.dot(x, self.W1)
        z_sig = np.tanh(z)  # Bottleneck activation
        x_hat = np.dot(z_sig, self.W2)
        
        # Online training (Gradient Descent on MSE)
        error = x_hat - x
        # Simplified gradient update for W2 and W1
        grad_W2 = np.outer(z_sig, error)
        grad_W1 = np.dot(x[:, None], (error @ self.W2.T * (1 - z_sig**2))[:, None]) # approximation
        
        self.W2 -= self.lr * grad_W2
        self.W1 -= self.lr * grad_W1.flatten()
        
        self.current_index += 1
        return {"price": price, "bottleneck": z_sig, "reconstruction_error": np.mean(error**2)}

    def get_signals(self):
        return []

    def get_params(self):
        return {"lr": self.lr, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim}

    def get_signal_dict(self, bottleneck, price):
        # Simple linear model on bottleneck: output = sum(bottleneck)
        output = np.sum(bottleneck)
        if output > 0.3:
            return {"type": "BUY", "price": float(price), "confidence": float(abs(output)), "metadata": {}}
        elif output < -0.3:
            return {"type": "SELL", "price": float(price), "confidence": float(abs(output)), "metadata": {}}
        return {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
