import numpy as np
import pandas as pd
import warnings

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class LightGBMSignal:
    """
    LightGBM-based trading strategy with walk-forward training.
    
    Fallback: Rolling logistic regression with numpy if LightGBM is unavailable.
    Features: RSI(14), MACD hist, BB%B, ATR%, volume_ratio(20), returns(1,3,5).
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.features = []
        self.model = None
        self.retrain_counter = 0
        self.training_window = 200
        self.last_train_idx = 0
        self.current_idx = 0
        self._prepare_features()

    def _prepare_features(self):
        df = self.data.copy()
        # RSI(14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD Histogram
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal_line
        # BB%B
        ma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_pct_b'] = (df['close'] - (ma - 2 * std)) / (4 * std)
        # ATR%
        high_low = df['high'] - df['low']
        high_cp = np.abs(df['high'] - df['close'].shift())
        low_cp = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        df['atr_pct'] = (atr / df['close']) * 100
        # volume_ratio
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        # returns
        df['ret1'] = df['close'].pct_change(1)
        df['ret3'] = df['close'].pct_change(3)
        df['ret5'] = df['close'].pct_change(5)
        
        self.df = df.fillna(0)
        self.feature_cols = ['rsi', 'macd_hist', 'bb_pct_b', 'atr_pct', 'vol_ratio', 'ret1', 'ret3', 'ret5']

    def _train_model(self, end_idx):
        start_idx = max(0, end_idx - self.training_window)
        train_df = self.df.iloc[start_idx:end_idx]
        if len(train_df) < self.training_window:
            return
        
        X = train_df[self.feature_cols].values
        y = (self.df['close'].shift(-1) > self.df['close']).iloc[start_idx:end_idx].values
        X, y = X[:-1], y[:-1]

        if LIGHTGBM_AVAILABLE:
            try:
                model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, verbose=-1)
                model.fit(X, y)
                self.model = model
            except:
                self._train_logistic_fallback(X, y)
        else:
            self._train_logistic_fallback(X, y)
        
        self.last_train_idx = end_idx

    def _train_logistic_fallback(self, X, y):
        """Rolling logistic regression using numpy."""
        n_features = X.shape[1]
        # X is (N, F), y is (N,)
        # Simple SGD-like approach to find weights
        weights = np.zeros(n_features)
        bias = 0.0
        lr = 0.01
        for _ in range(100):
            for i in range(len(X)):
                z = np.dot(X[i], weights) + bias
                pred = 1.0 / (1.0 + np.exp(-z))
                error = y[i] - pred
                weights += lr * error * X[i]
                bias += lr * error
        self.model = {"weights": weights, "bias": bias}

    def _predict(self, features):
        if self.model is None:
            return 0.5
        
        if LIGHTGBM_AVAILABLE and not isinstance(self.model, dict):
            try:
                return self.model.predict_proba(features.reshape(1, -1))[0][1]
            except:
                return 0.5
        else:
            # Logistic fallback
            w = self.model["weights"]
            b = self.model["bias"]
            z = np.dot(features, w) + b
            return 1.0 / (1.0 + np.exp(-z))

    def init(self, data):
        self.data = data
        self._prepare_features()
        return self

    def next(self, bar_idx):
        if (bar_idx - self.last_train_idx) >= 50 or self.model is None:
            if bar_idx >= self.training_window:
                self._train_model(bar_idx)
        
        features = self.df.iloc[bar_idx][self.feature_cols].values
        prob = self._predict(features)
        price = self.data.iloc[bar_idx]['close']
        
        signal_type = "HOLD"
        if prob > 0.65: signal_type = "BUY"
        elif prob < 0.35: signal_type = "SELL"
            
        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(prob),
            "metadata": {"model": "lightgbm" if LIGHTGBM_AVAILABLE and not isinstance(self.model, dict) else "logistic_reg"}
        }

    def get_signals(self): return []
    def get_params(self): return {"features": self.feature_cols}

if __name__ == "__main__":
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 105,
        'low': np.random.randn(500).cumsum() + 95,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.rand(500) * 1000
    })
    strat = LightGBMSignal(df).init(df)
    for i in range(250, 260):
        print(f"Idx {i}: {strat.next(i)}")
