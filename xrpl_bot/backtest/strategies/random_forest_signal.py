import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class RandomForestSignal:
    \"\"\"
    Random Forest Signal strategy.
    Features: RSI, MACD, BB%B, ATR%, vol_ratio, returns.
    Uses sklearn RandomForest (100 estimators) if available.
    Confidence = prediction probability.
    Walk-forward implementation.
    \"\"\"

    def __init__(self, data):
        self.data = data
        self.current_index = 0
        self.model = None
        self.features_history = []
        self.lookback = 50  # Minimum bars to start training
        
        # Pre-calculate indicators for the entire dataset to allow walk-forward simulation
        # Note: In a real live env, we'd compute this incrementally.
        self.features_df = self._compute_features(data)
        self.labels = self._compute_labels(data)
        
        self.signals = []

    def _compute_features(self, data):
        # data is expected to have close, high, low, volume
        # Returns a 2D numpy array of features
        close = data['close'].values if hasattr(data, 'values') else np.array(data['close'])
        high = data['high'].values if hasattr(data, 'values') else np.array(data['high'])
        low = data['low'].values if hasattr(data, 'values') else np.array(data['low'])
        vol = data['volume'].values if hasattr(data, 'values') else np.array(data['volume'])
        
        n = len(close)
        features = np.zeros((n, 6))
        
        # 1. Returns
        returns = np.diff(close, prepend=close[0])
        
        # 2. RSI (14)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = self._ema(gain, 14)
        avg_loss = self._ema(loss, 14)
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # 3. MACD (12, 26, 9)
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd = ema12 - ema26
        signal_line = self._ema(macd, 9)
        macd_hist = macd - signal_line
        
        # 4. Bollinger Bands %B (20, 2)
        ma20 = self._ema(close, 20)
        std20 = self._rolling_std(close, 20)
        upper = ma20 + (2 * std20)
        lower = ma20 - (2 * std20)
        bb_pct_b = (close - lower) / (upper - lower + 1e-9)
        
        # 5. ATR% (14)
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        atr = self._ema(tr, 14)
        atr_pct = (atr / close) * 100
        
        # 6. Vol Ratio (vol / avg_vol_20)
        avg_vol = self._ema(vol, 20)
        vol_ratio = vol / (avg_vol + 1e-9)

        for i in range(n):
            features[i] = [
                rsi[i],
                macd_hist[i],
                bb_pct_b[i],
                atr_pct[i],
                vol_ratio[i],
                returns[i]
            ]
        return features

    def _compute_labels(self, data):
        # Label 1 if returns in next 3 bars > 0, else 0
        close = data['close'].values if hasattr(data, 'values') else np.array(data['close'])
        labels = np.zeros(len(close))
        for i in range(len(close) - 3):
            if close[i+3] > close[i]:
                labels[i] = 1
        return labels

    def _ema(self, series, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

    def _rolling_std(self, series, period):
        std = np.zeros_like(series)
        for i in range(len(series)):
            if i < period:
                std[i] = 0
            else:
                std[i] = np.std(series[i-period:i+1])
        return std

    def next(self, bar):
        idx = self.current_index
        price = bar['close']
        
        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
        
        if SKLEARN_AVAILABLE and idx >= self.lookback:
            # Walk-forward: Train on all data up to current index
            X_train = self.features_df[:idx]
            y_train = self.labels[:idx]
            
            # We only train if we have a decent sample
            if len(X_train) > 20:
                if self.model is None or idx % 10 == 0: # Retrain every 10 bars
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.model.fit(X_train, y_train)
                
                if self.model:
                    current_feat = self.features_df[idx].reshape(1, -1)
                    prob = self.model.predict_proba(current_feat)[0]
                    # prob[1] is the probability of class 1 (upward movement)
                    conf = float(prob[1])
                    
                    if conf > 0.6:
                        signal = {
                            "type": "BUY",
                            "price": float(price),
                            "confidence": conf,
                            "metadata": {"prob": conf}
                        }
                    elif conf < 0.4:
                        signal = {
                            "type": "SELL",
                            "price": float(price),
                            "confidence": 1.0 - conf,
                            "metadata": {"prob": conf}
                        }

        self.current_index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"n_estimators": 100, "sklearn_available": SKLEARN_AVAILABLE}
