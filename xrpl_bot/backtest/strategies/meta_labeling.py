import numpy as np
import pandas as pd

class MetaLabelingStrategy:
    """
    Meta-Labeling Strategy.
    Layer 1: Rule-based (EMA Crossover: fast=9, slow=21).
    Layer 2: ML Filter (Logistic) deciding whether to take Layer 1 signal.
    Features: Trend strength (ADX-like), Volatility regime (ATR rank), Volume confirmation.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.df = data.copy()
        self.model_weights = None  # Will be trained weights
        self.training_window = 200
        self.last_train_idx = 0
        self.current_idx = 0
        self._prepare_indicators()

    def _prepare_indicators(self):
        df = self.df
        # EMA Crossover
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Trend Strength (Simplified ADX: +DI - -DI)
        up = df['high'].diff().clip(lower=0)
        down = (-df['low'].diff().clip(upper=0))
        df['plus_di'] = up.rolling(14).mean()
        df['minus_di'] = down.rolling(14).mean()
        df['trend_strength'] = (df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9)

        # Volatility Regime (ATR Rank)
        high_low = df['high'] - df['low']
        df['atr'] = high_low.rolling(14).mean()
        df['volatility_regime'] = df['atr'].rolling(50).rank(pct=True)

        # Volume Confirmation
        df['vol_confirm'] = df['volume'] / df['volume'].rolling(20).mean()

        # Target for Layer 2: 1 if Layer 1 signal was correct, else 0
        # A signal is correct if price goes in the direction of the EMA crossover within 3 bars
        df['l1_signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'l1_signal'] = 1
        df.loc[df['ema_fast'] < df['ema_slow'], 'l1_signal'] = -1
        
        # Correctness: did price move up in direction of BUY (1) or down in direction of SELL (-1)
        # We look ahead 3 bars
        future_returns = df['close'].shift(-3) / df['close'] - 1
        df['is_correct'] = 0
        df.loc[(df['l1_signal'] == 1) & (future_returns > 0), 'is_correct'] = 1
        df.loc[(df['l1_signal'] == -1) & (future_returns < 0), 'is_correct'] = 1

        self.df = df.fillna(0)
        self.feature_cols = ['trend_strength', 'volatility_regime', 'vol_confirm']

    def _train_layer2(self, end_idx):
        start_idx = max(0, end_idx - self.training_window)
        train_df = self.df.iloc[start_idx:end_idx]
        
        if len(train_df) < 50:
            return
        
        # Only train on rows where Layer 1 actually produced a signal
        l1_active = train_df[train_df['l1_signal'] != 0]
        if len(l1_active) < 20:
            return
            
        X = l1_active[self.feature_cols].values
        y = l1_active['is_correct'].values

        # Simple Logistic Regression via Gradient Descent (no sklearn)
        n_features = X.shape[1]
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
        
        self.model_weights = {"w": weights, "b": bias}
        self.last_train_idx = end_idx

    def _predict_layer2(self, features):
        if self.model_weights is None:
            return 0.5 # Neutral filter
        
        w = self.model_weights["w"]
        b = self.model_weights["b"]
        z = np.dot(features, w) + b
        return 1.0 / (1.0 + np.exp(-z))

    def init(self, data):
        self.data = data
        self._prepare_indicators()
        return self

    def next(self, bar_idx):
        if (bar_idx - self.last_train_idx) >= 50 or self.model_weights is None:
            if bar_idx >= self.training_window:
                self._train_layer2(bar_idx)
        
        row = self.df.iloc[bar_idx]
        l1_sig = row['l1_signal']
        
        # Layer 2 Filter
        features = row[self.feature_cols].values
        l2_agreement = self._predict_layer2(features)
        
        price = self.data.iloc[bar_idx]['close']
        signal_type = "HOLD"
        confidence = 0.0
        
        if l1_sig == 1 and l2_agreement > 0.5:
            signal_type = "BUY"
            # Layer 1 confidence is just 1.0 (rule-based)
            confidence = 1.0 * l2_agreement
        elif l1_sig == -1 and l2_agreement > 0.5:
            signal_type = "SELL"
            confidence = 1.0 * l2_agreement
            
        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {"l2_prob": float(l2_agreement), "l1_sig": int(l1_sig)}
        }

    def get_signals(self): return []
    def get_params(self): return {"l1_fast": 9, "l1_slow": 21}

if __name__ == "__main__":
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 105,
        'low': np.random.randn(500).cumsum() + 95,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.rand(500) * 1000
    })
    strat = MetaLabelingStrategy(df).init(df)
    for i in range(250, 260):
        print(f"Idx {i}: {strat.next(i)}")
