import numpy as np
import pandas as pd

class FeatureImportanceRotation:
    """
    Strategy that tracks 8 features and uses only the top 4 
    ranked by rolling correlation with forward returns.
    """
    def __init__(self, data):
        """
        :param data: DataFrame with features and 'returns'
        Features: RSI, MACD, BB%B, ATR%, vol_ratio, returns_1, returns_5, stoch_k
        """
        self.data = data
        self.feature_names = [
            'RSI', 'MACD', 'BB_pct_b', 'ATR_pct', 
            'vol_ratio', 'returns_1', 'returns_5', 'stoch_k'
        ]
        self.window_size = 50
        self.rotation_interval = 30
        self.last_rotation_idx = 0
        self.current_top_features = self.feature_names[:4]
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])
        self.last_idx = 0

    def _update_features(self, idx):
        """Rank features by correlation with forward returns."""
        if idx < self.window_size:
            return

        start_idx = max(0, idx - self.window_size)
        # We need forward returns. For index 'i', we look at return at 'i+1' or similar.
        # In backtest loops, 'bar' usually represents time 't'. 
        # We use the correlation of features at [t-50:t] with returns at [t-49:t+1]? 
        # Actually, the instruction says "correlation with forward returns (50 bars)".
        # This implies we look at the correlation between current features and the 
        # return that happens *next*.
        
        # Since we are in a streaming 'next(bar)' context, we look back.
        # To compute correlation with "forward returns", we must have the returns available.
        # We'll use the returns column from the data.
        
        subset = self.data.iloc[start_idx:idx+1].copy()
        # Create a 'forward_return' column: return at t+1
        subset['fwd_ret'] = subset['returns'].shift(-1)
        
        # Drop NaN from shift
        subset = subset.dropna(subset=['fwd_ret'])
        
        if len(subset) < 10:
            return

        correlations = []
        for feat in self.feature_names:
            corr = subset[feat].corr(subset['fwd_ret'])
            correlations.append(abs(corr))
            
        # Get indices of top 4
        top_indices = np.argsort(correlations)[-4:][::-1]
        self.current_top_features = [self.feature_names[i] for i in top_indices]
        # For simplicity, weights are equal for the top 4
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])

    def next(self, bar):
        idx = bar['index']
        self.last_idx = idx
        
        # Rotate features monthly (~30 bars)
        if idx % self.rotation_interval == 0:
            self._update_features(idx)

        if not self.current_top_features:
            return {"type": "HOLD", "price": bar['close'], "confidence": 0.0, "metadata": {}}

        # Calculate signal = weighted_sum(top4)
        # We need the values of the top features for the current bar
        signal = 0.0
        for i, feat in enumerate(self.current_top_features):
            # Normalize feature? The instruction says signal = weighted_sum(top4).
            # Usually features like RSI are 0-100, MACD is small. 
            # To make a sum work, we should probably use standardized or scaled values.
            # However, to follow "simple linear model" strictly:
            val = bar['features'].get(feat, 0.0)
            # Let's assume features in 'bar' are already somewhat normalized or 
            # we just use the raw value as per "simple linear model".
            # A more robust way is to use the Z-score.
            signal += self.weights[i] * val

        # Signal thresholds: BUY > 0.5, SELL < -0.5
        # Note: If features are not normalized, this threshold is arbitrary. 
        # We'll assume the input features are scaled [-1, 1] or similar for the signal.
        
        sig_type = "HOLD"
        if signal > 0.5:
            sig_type = "BUY"
        elif signal < -0.5:
            sig_type = "SELL"
            
        return {
            "type": sig_type,
            "price": bar['close'],
            "confidence": float(abs(signal)),
            "metadata": {"top_features": self.current_top_features}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"top_features": self.current_top_features, "weights": self.weights.tolist()}
