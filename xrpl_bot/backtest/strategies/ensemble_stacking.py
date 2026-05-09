import numpy as np
import pandas as pd
from xrpl_bot.backtest.strategies.base_strategy import BaseStrategy, Signal

class EnsembleStacking(BaseStrategy):
    """
    Ensemble Stacking Strategy using 3 base models.
    Base models: SMA crossover, RSI signal, Momentum signal.
    Meta-learner: Weighted average optimized by rolling performance.
    """

    def __init__(self, params=None):
        super().__init__()
        self.params = params or {}
        self.weights = np.array([0.33, 0.33, 0.34])
        self.performance_history = [] 
        self.lookback_performance = 50
        self.current_idx = 0
        self.last_signals = None
        self.df = None

    def init(self, data):
        """
        data: pd.DataFrame with columns ['close', 'high', 'low', 'open', 'volume']
        """
        self.data = data
        self.df = data.copy()
        self._prepare_indicators()
        self.signals = []
        return self

    def _prepare_indicators(self):
        df = self.df
        # SMA
        df['sma_fast'] = df['close'].rolling(window=10).mean()
        df['sma_slow'] = df['close'].rolling(window=30).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan).fillna(1))))
        
        # Momentum (5-period)
        df['momentum'] = df['close'].pct_change(5)
        self.df = df.fillna(0)

    def _base_model_sma(self, idx):
        row = self.df.iloc[idx]
        if row['sma_fast'] > row['sma_slow']:
            return 0.8 # Strong signal
        elif row['sma_fast'] < row['sma_slow']:
            return 0.2 # Weak signal
        return 0.5

    def _base_model_rsi(self, idx):
        row = self.df.iloc[idx]
        if row['rsi'] < 30:
            return 0.8
        elif row['rsi'] > 70:
            return 0.2
        return 0.5

    def _base_model_momentum(self, idx):
        row = self.df.iloc[idx]
        if row['momentum'] > 0.01:
            return 0.8
        elif row['momentum'] < -0.01:
            return 0.2
        return 0.5

    def next(self, bar_idx: int) -> Signal:
        self.current_idx = bar_idx
        
        # 1. Get base signals
        s1 = self._base_model_sma(bar_idx)
        s2 = self._base_model_rsi(bar_idx)
        s3 = self._base_model_momentum(bar_idx)
        base_signals = np.array([s1, s2, s3])

        # 2. Meta-learner: Weighted Average
        ensemble_score = np.dot(self.weights, base_signals)

        # 3. Track performance for optimization (on previous bar's prediction)
        if bar_idx > 0 and len(self.performance_history) > 0 and self.last_signals is not None:
            prev_idx = bar_idx - 1
            # Check if prediction at prev_idx led to a good return at current_idx
            try:
                actual_ret = (self.df.iloc[bar_idx]['close'] / self.df.iloc[prev_idx]['close']) - 1
            except:
                actual_ret = 0.0
            
            self.performance_history.append({
                "signals": self.last_signals,
                "return": actual_ret
            })
            
            if len(self.performance_history) > self.lookback_performance:
                self.performance_history.pop(0)
                
            if len(self.performance_history) >= self.lookback_performance:
                self._optimize_weights()

        self.last_signals = base_signals
        try:
            price = float(self.df.iloc[bar_idx]['close'])
        except:
            price = 0.0
        
        # Try to get timestamp from index
        try:
            timestamp = float(self.df.index[bar_idx])
        except:
            timestamp = float(bar_idx)
        
        signal_type = "HOLD"
        if ensemble_score > 0.6:
            signal_type = "BUY"
        elif ensemble_score < 0.4:
            signal_type = "SELL"

        sig = Signal(
            type=signal_type,
            price=price,
            timestamp=timestamp,
            confidence=float(ensemble_score),
            metadata={"weights": self.weights.tolist()}
        )
        self.signals.append(sig)
        return sig

    def _optimize_weights(self):
        scores = np.zeros(3)
        for entry in self.performance_history:
            ret = entry['return']
            sigs = entry['signals']
            for i in range(3):
                if ret > 0:
                    scores[i] += sigs[i]
                else:
                    scores[i] += (1.0 - sigs[i])
        
        total = np.sum(scores)
        if total > 0:
            self.weights = scores / total
        else:
            self.weights = np.array([0.33, 0.33, 0.34])

    def get_signals(self) -> list[Signal]:
        return self.signals

    def get_params(self) -> dict:
        return {"base_models": ["SMA", "RSI", "Momentum"], "lookback": self.lookback_performance}
