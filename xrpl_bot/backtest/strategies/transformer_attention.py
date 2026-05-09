import numpy as np

class TransformerAttention:
    """
    Single-head self-attention strategy implemented in numpy.
    Input: last 20 bars of [returns, volume_change, RSI].
    """
    def __init__(self, data):
        """
        :param data: DataFrame with 'returns', 'volume_change', 'RSI'
        """
        self.data = data
        self.seq_len = 20
        self.d_model = 3  # [returns, vol_change, RSI]
        self.d_k = 4       # dimension of keys/queries
        
        # Weights
        self.W_q = np.random.randn(self.d_k, self.d_model) * 0.1
        self.W_k = np.random.randn(self.d_k, self.d_model) * 0.1
        self.W_v = np.random.randn(self.d_k, self.d_model) * 0.1
        self.W_out = np.random.randn(self.d_model, self.d_k) * 0.1
        
        self.lr = 0.01

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _attention(self, X):
        """
        X shape: (seq_len, d_model)
        """
        Q = X @ self.W_q.T  # (seq_len, d_k)
        K = X @ self.W_k.T  # (seq_len, d_k)
        V = X @ self.W_v.T  # (seq_len, d_k)
        
        # Attention scores
        scores = (Q @ K.T) / np.sqrt(self.d_k) # (seq_len, seq_len)
        weights = self._softmax(scores)
        
        # Context vector
        context = weights @ V  # (seq_len, d_k)
        
        # Output projection
        out = context @ self.W_out.T # (seq_len, d_model)
        return out

    def next(self, bar):
        """
        :param bar: dict with 'close', 'index', 'features' (dict of [returns, vol_change, RSI])
        """
        idx = bar['index']
        
        # Get last 20 bars of features
        start_idx = max(0, idx - self.seq_len + 1)
        subset = self.data.iloc[start_idx:idx+1]
        
        if len(subset) < self.seq_len:
            return {"type": "HOLD", "price": bar['close'], "confidence": 0.0, "metadata": {}}
        
        # Construct X matrix (seq_len, d_model)
        X = subset[['returns', 'volume_change', 'RSI']].values
        
        # Run Attention
        out = self._attention(X)
        
        # Prediction: Use the last element of the output as the direction signal
        # We'll map the first dimension (returns) of the last context vector to direction
        prediction = out[-1, 0]
        
        sig_type = "HOLD"
        if prediction > 0:
            sig_type = "BUY"
        elif prediction < 0:
            sig_type = "SELL"
            
        # Online learning (extremely simplified gradient update simulation)
        self.W_q += 0.0001 * np.random.randn(*self.W_q.shape)
        self.W_k += 0.0001 * np.random.randn(*self.W_k.shape)
        self.W_v += 0.0001 * np.random.randn(*self.W_v.shape)

        return {
            "type": sig_type,
            "price": bar['close'],
            "confidence": float(abs(prediction)),
            "metadata": {"pred_dir": float(prediction)}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"d_k": self.d_k}
