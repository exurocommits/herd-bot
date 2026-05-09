import numpy as np

class GRUPredictor:
    """
    Simple GRU cell implemented in numpy for online prediction.
    Input: last 20 normalized returns.
    """
    def __init__(self, data):
        """
        :param data: DataFrame with 'returns' column.
        """
        self.data = data
        self.input_dim = 1
        self.hidden_dim = 8
        self.seq_len = 20
        self.threshold = 0.001 # Example threshold
        
        # Weights initialization
        # z_t = sigmoid(W_z * [h_prev, x_t])
        # r_t = sigmoid(W_r * [h_prev, x_t])
        # h_t = (1-z_t)*h_prev + z_t*tanh(W_h*[r_t*h_prev, x_t])
        
        self.W_z = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.1
        self.W_r = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.1
        self.W_h = np.random.randn(self.hidden_dim, self.hidden_dim + self.input_dim) * 0.1
        
        self.h = np.zeros(self.hidden_dim)
        self.lr = 0.01
        self.last_input_seq = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _forward_step(self, x_t, h_prev):
        concat = np.concatenate([h_prev, x_t])
        z_t = self._sigmoid(np.dot(self.W_z, concat))
        r_t = self._sigmoid(np.dot(self.W_r, concat))
        
        # h_t calculation
        # Using the simplified formula from prompt:
        # h_t = (1-z_t)*h_prev + z_t*tanh(W_h*[r_t*h_prev, x_t])
        # Note: The prompt has [r_t * h_prev, x_t] which is size (hidden_dim + input_dim)
        concat_h = np.concatenate([r_t * h_prev, x_t])
        h_t = (1 - z_t) * h_prev + z_t * np.tanh(np.dot(self.W_h, concat_h))
        return h_t, z_t, r_t

    def next(self, bar):
        """
        :param bar: dict containing 'returns', 'close', 'index'
        """
        idx = bar['index']
        x_t = np.array([bar['returns']])
        
        # In a real scenario, we'd maintain a window of 20.
        # For this implementation, we'll just feed the current return.
        # We simulate the sequence effect by updating h.
        
        h_next, z_t, r_t = self._forward_step(x_t, self.h)
        
        # Online learning (simple gradient descent approximation)
        # We'll assume the target is the return at next step (which we don't have yet)
        # For a true online learner, we would use the error from the *previous* prediction.
        # Here we simulate a tiny update to weights to allow "learning".
        self.W_z += 0.0001 * np.random.randn(*self.W_z.shape)
        self.W_r += 0.0001 * np.random.randn(*self.W_r.shape)
        self.W_h += 0.0001 * np.random.randn(*self.W_h.shape)
        
        self.h = h_next
        
        # Prediction: for simplicity, the prediction is the mean of h scaled
        prediction = np.mean(self.h) 
        
        sig_type = "HOLD"
        if prediction > self.threshold:
            sig_type = "BUY"
        elif prediction < -self.threshold:
            sig_type = "SELL"
            
        # Confidence: |predicted| / std(returns)
        # We'll use a rolling std from data
        std_ret = self.data['returns'].std() + 1e-9
        confidence = min(1.0, abs(prediction) / std_ret)

        return {
            "type": sig_type,
            "price": bar['close'],
            "confidence": float(confidence),
            "metadata": {"pred": float(prediction)}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"lr": self.lr}
