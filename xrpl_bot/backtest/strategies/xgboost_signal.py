import numpy as np
import pandas as pd
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class XGBoostSignal:
    """
    XGBoost-based trading strategy with walk-forward training and a decision stump fallback.
    
    Features: RSI(14), MACD hist, BB%B, ATR%, volume_ratio(20), returns(1,3,5).
    Fallback: Decision stump (threshold-based) if xgboost is unavailable.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the strategy.
        :param data: DataFrame containing 'open', 'high', 'low', 'close', 'volume'.
        """
        self.data = data
        self.features = []
        self.model = None
        self.retrain_counter = 0
        self.training_window = 200
        self.retrain_interval = 50
        self.last_train_idx = 0
        
        # State for walk-forward
        self.current_idx = 0
        self._prepare_features()

    def _prepare_features(self):
        """Compute technical indicators used for feature engineering."""
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
        
        # BB%B (Bollinger Band %B)
        ma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_pct_b'] = (df['close'] - (ma - 2 * std)) / (4 * std)
        
        # ATR% (Average True Range %)
        high_low = df['high'] - df['low']
        high_cp = np.abs(df['high'] - df['close'].shift())
        low_cp = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        df['atr_pct'] = (atr / df['close']) * 100
        
        # volume_ratio(20)
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # returns(1,3,5)
        df['ret1'] = df['close'].pct_change(1)
        df['ret3'] = df['close'].pct_change(3)
        df['ret5'] = df['close'].pct_change(5)
        
        self.df = df.fillna(0)
        self.feature_cols = ['rsi', 'macd_hist', 'bb_pct_b', 'atr_pct', 'vol_ratio', 'ret1', 'ret3', 'ret5']

    def _get_training_set(self, end_idx):
        """Extract training features and labels."""
        start_idx = max(0, end_idx - self.training_window)
        train_df = self.df.iloc[start_idx:end_idx]
        
        if len(train_df) < self.training_window:
            return None, None
        
        X = train_df[self.feature_cols].values
        # Label: 1 if next close > current close, else 0
        y = (self.df['close'].shift(-1) > self.df['close']).iloc[start_idx:end_idx].values
        
        # Remove last one because shift(-1) is NaN at the end
        return X[:-1], y[:-1]

    def _train_model(self, end_idx):
        """Train XGBoost or fallback to decision stump."""
        X, y = self._get_training_set(end_idx)
        if X is None or len(X) == 0:
            return

        if XGBOOST_AVAILABLE:
            try:
                model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
                model.fit(X, y)
                self.model = model
            except Exception:
                self._train_fallback(X, y)
        else:
            self._train_fallback(X, y)
            
        self.last_train_idx = end_idx

    def _train_fallback(self, X, y):
        """Simple decision stump fallback: if mean return > 0, signal 1."""
        # We store a simple threshold of the first feature (e.g., RSI)
        # or just a weight vector for a linear stub.
        self.model = "stump"
        self.stump_threshold = np.mean(X[:, 0]) # Threshold on RSI
        self.stump_direction = np.mean(y) # Bias

    def _predict(self, features):
        """Predict probability using trained model."""
        if self.model is None:
            return 0.5
        
        if XGBOOST_AVAILABLE and not isinstance(self.model, str):
            try:
                # prob of class 1
                return self.model.predict_proba(features.reshape(1, -1))[0][1]
            except:
                return 0.5
        else:
            # Fallback prediction
            # If RSI > threshold and bias > 0.5 -> prob high
            rsi_val = features[0]
            if rsi_val > self.stump_threshold:
                return min(0.8, self.stump_direction + 0.2)
            else:
                return max(0.2, self.stump_direction - 0.2)

    def init(self, data):
        self.data = data
        self._prepare_features()
        return self

    def next(self, bar_idx):
        """
        Process next bar.
        :param bar_idx: current index in the dataframe.
        :return: dict with signal info.
        """
        self.current_idx = bar_idx
        
        # Check for retraining
        if (bar_idx - self.last_train_idx) >= self.retrain_interval or self.model is None:
            if bar_idx >= self.training_window:
                self._train_model(bar_idx)
        
        # Predict
        features = self.df.iloc[bar_idx][self.feature_cols].values
        prob = self._predict(features)
        
        price = self.data.iloc[bar_idx]['close']
        
        signal_type = "HOLD"
        if prob > 0.65:
            signal_type = "BUY"
        elif prob < 0.35:
            signal_type = "SELL"
            
        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(prob),
            "metadata": {"model": "xgboost" if XGBOOST_AVAILABLE and not isinstance(self.model, str) else "stump"}
        }

    def get_signals(self):
        """Not used in real-time next() flow but required by interface."""
        return []

    def get_params(self):
        """Return strategy parameters."""
        return {
            "training_window": self.training_window,
            "retrain_interval": self.retrain_interval,
            "features": self.feature_cols
        }

if __name__ == "__main__":
    # Simple test
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 105,
        'low': np.random.randn(500).cumsum() + 95,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.rand(500) * 1000
    })
    strat = XGBoostSignal(df).init(df)
    for i in range(250, 300):
        print(f"Idx {i}: {strat.next(i)}")
