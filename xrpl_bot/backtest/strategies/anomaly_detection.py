import numpy as np
import pandas as pd

class AnomalyDetectionSignal:
    """
    Anomaly Detection Strategy.
    Detects price/volume anomalies using Mahalanobis distance.
    Features: return, volume_change, ATR_ratio, RSI.
    Implementation: Mahalanobis distance using rolling mean and covariance (50-bar).
    No sklearn dependency.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.df = data.copy()
        self.window = 50
        self._prepare_features()

    def _prepare_features(self):
        df = self.df
        # Returns
        df['ret'] = df['close'].pct_change()
        # Volume change
        df['vol_change'] = df['volume'].pct_change()
        # ATR Ratio
        high_low = df['high'] - df['low']
        df['atr'] = high_low.rolling(14).mean()
        df['atr_ratio'] = high_low / (df['atr'] + 1e-9)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        self.df = df.fillna(0)
        self.feature_cols = ['ret', 'vol_change', 'atr_ratio', 'rsi']

    def _calculate_mahalanobis(self, features_subset):
        """
        Manually calculate Mahalanobis distance for a single point given a distribution.
        """
        if len(features_subset) < self.window:
            return 0.0
        
        # X is (N, F)
        X = features_subset.values
        # Target is the last row
        target = X[-1]
        # Distribution is all rows
        mu = np.mean(X, axis=0)
        try:
            cov = np.cov(X, rowvar=False)
            inv_cov = np.linalg.inv(cov + np.eye(X.shape[1]) * 1e-6) # Regularized
            diff = target - mu
            dist_sq = diff.T @ inv_cov @ diff
            return np.sqrt(dist_sq)
        except np.linalg.LinAlgError:
            return 0.0

    def init(self, data):
        self.data = data
        self._prepare_features()
        return self

    def next(self, bar_idx):
        if bar_idx < self.window:
            return {"type": "HOLD", "price": float(self.data.iloc[bar_idx]['close']), "confidence": 0.0, "metadata": {}}

        # Get the window of features ending at bar_idx
        start_idx = max(0, bar_idx - self.window + 1)
        features_window = self.df.iloc[start_idx : bar_idx + 1][self.feature_cols]
        
        dist = self._calculate_mahalanobis(features_window)
        
        price = self.data.iloc[bar_idx]['close']
        atr = self.df.iloc[bar_idx]['atr']
        ret = self.df.iloc[bar_idx]['ret']
        
        signal_type = "HOLD"
        confidence = min(dist / 3.0, 1.0)
        
        if dist > 3.0:
            # Anomaly detected
            if ret < -2 * (atr / self.data.iloc[bar_idx]['close'] if atr > 0 else 0.01):
                # Oversold anomaly (simplified check)
                # Using raw ret vs ATR as requested
                pass # Logic below is cleaner
            
            # Re-implementing specific prompt logic:
            # "BUY if return < -2*ATR (oversold anomaly), SELL if return > +2*ATR"
            # Note: return is pct, ATR is absolute. 
            # For consistency, we'll treat 'return' in the prompt as the actual price change or normalized.
            # Given the context, we'll assume return is the pct change and compare to normalized ATR.
            norm_atr = (atr / self.data.iloc[bar_idx]['close']) if self.data.iloc[bar_idx]['close'] != 0 else 0
            
            if ret < -2 * norm_atr:
                signal_type = "BUY"
            elif ret > 2 * norm_atr:
                signal_type = "SELL"

        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {"mahalanobis_dist": float(dist)}
        }

    def get_signals(self): return []
    def get_params(self): return {"window": self.window, "features": self.feature_cols}

if __name__ == "__main__":
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 105,
        'low': np.random.randn(500).cumsum() + 95,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.rand(500) * 1000
    })
    # Inject an anomaly
    df.iloc[400:405, df.columns.get_loc('close')] *= 0.8 
    
    strat = AnomalyDetectionSignal(df).init(df)
    for i in range(350, 410):
        print(f"Idx {i}: {strat.next(i)}")
