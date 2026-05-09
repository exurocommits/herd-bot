"""
Top-5 Ensemble Strategy — Pure Python, no numpy.
Combines the 5 best pure-Python strategies using a weighted voting system.
Weights are pre-tuned based on typical performance characteristics:
  - EMA Cross (30%): trend following, good for momentum
  - Supertrend (25%): trend following, filters noise
  - MACD Momentum (20%): momentum confirmation
  - Bollinger Reversion (15%): mean reversion in ranges
  - Stoch+RSI (10%): reversal confirmation
  
The ensemble uses a consensus threshold to avoid whipsaws.
"""
import math
from typing import List, Optional, Dict, Any
from .base_strategy import BaseStrategy, Signal
from .ema_cross_pure import EMACrossPure
from .bollinger_reversion_pure import BollingerReversionPure
from .supertrend_pure import SupertrendPure
from .macd_momentum_pure import MACDMomentumPure
from .stoch_rsi_confluence import StochRSIConfluence


class Top5Ensemble(BaseStrategy):
    """
    Weighted ensemble of 5 pure-Python strategies.
    Uses confidence-weighted voting with a consensus threshold.
    """
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        buy_threshold: float = 0.35,
        sell_threshold: float = 0.35,
        # Sub-strategy params (pass-through)
        ema_fast: int = 9, ema_slow: int = 21,
        bb_period: int = 20, bb_std: float = 2.0,
        st_period: int = 10, st_mult: float = 3.0,
        macd_fast: int = 12, macd_slow: int = 26, macd_sig: int = 9,
        stoch_k: int = 14, stoch_d: int = 3,
    ):
        super().__init__()
        
        self.strategies = [
            EMACrossPure(fast_period=ema_fast, slow_period=ema_slow),
            SupertrendPure(period=st_period, multiplier=st_mult),
            MACDMomentumPure(fast=macd_fast, slow=macd_slow, signal_period=macd_sig),
            BollingerReversionPure(bb_period=bb_period, bb_std=bb_std),
            StochRSIConfluence(stoch_k=stoch_k, stoch_d=stoch_d),
        ]
        
        self.weights = weights or [0.30, 0.25, 0.20, 0.15, 0.10]
        if len(self.weights) != len(self.strategies):
            raise ValueError(f"Weights length ({len(self.weights)}) must match strategies ({len(self.strategies)})")
        
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._signal_history: List[Dict] = []

    def init(self, data):
        self.data = data
        self.signals = []
        for strat in self.strategies:
            strat.init(data)

    def next(self, bar_idx: int) -> Signal:
        price = self.data['close'][bar_idx]
        ts = self.data['timestamp'][bar_idx]
        
        # Collect weighted votes
        buy_score = 0.0
        sell_score = 0.0
        strategy_details = []
        
        for i, strategy in enumerate(self.strategies):
            try:
                sig = strategy.next(bar_idx)
            except Exception:
                sig = Signal('HOLD', price, ts, 0.0)
            
            weight = self.weights[i]
            detail = {
                'name': strategy.__class__.__name__,
                'type': sig.type,
                'confidence': sig.confidence,
                'weight': weight,
                'weighted_score': 0.0
            }
            
            if sig.type == 'BUY':
                score = sig.confidence * weight
                buy_score += score
                detail['weighted_score'] = score
            elif sig.type == 'SELL':
                score = sig.confidence * weight
                sell_score += score
                detail['weighted_score'] = -score
            
            strategy_details.append(detail)
        
        # Decision with threshold
        signal_type = 'HOLD'
        confidence = 0.0
        
        if buy_score > self.buy_threshold and buy_score > sell_score:
            signal_type = 'BUY'
            confidence = min(1.0, buy_score)
        elif sell_score > self.sell_threshold and sell_score > buy_score:
            signal_type = 'SELL'
            confidence = min(1.0, sell_score)
        
        # Record ensemble state
        self._signal_history.append({
            'bar': bar_idx,
            'buy_score': round(buy_score, 4),
            'sell_score': round(sell_score, 4),
            'details': strategy_details
        })
        
        # Keep history bounded
        if len(self._signal_history) > 200:
            self._signal_history = self._signal_history[-100:]
        
        sig = Signal(signal_type, price, ts, confidence,
                     metadata={
                         'buy_score': round(buy_score, 4),
                         'sell_score': round(sell_score, 4),
                         'strategies': {d['name']: d['type'] for d in strategy_details}
                     })
        self.signals.append(sig)
        return sig

    def get_params(self) -> dict:
        return {
            'weights': self.weights,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'strategies': [s.get_params() for s in self.strategies]
        }

    def get_ensemble_state(self) -> Dict:
        """Return current ensemble state for monitoring."""
        if not self._signal_history:
            return {}
        return self._signal_history[-1]
