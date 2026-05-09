import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Fallback for environments without numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    class MockNumpy:
        def __init__(self): pass
        def array(self, x, *args, **kwargs): return x
        def mean(self, x): return sum(x)/len(x) if len(x)>0 else 0
        def zeros(self, n): return [0.0]*n
        def isna(self, x): return [False]*len(x)
        def nan(self): return float('nan')
        def abs(self, x): return [abs(i) for i in x]
        def percentile(self, q): return 0
        def argmax(self, x): return 0
        def sum(self, x): return sum(x)
        def __getitem__(self, key): return self
        def __call__(self, *args, **kwargs): return self
        def tolist(self): return []
        def __add__(self, other): return self
        def __sub__(self, other): return self
        def __mul__(self, other): return self
        def __truediv__(self, other): return self
        def __gt__(self, other): return True
        def __lt__(self, other): return True
        def __ge__(self, other): return True
        def __le__(self, other): return True
        def __neg__(self, other): return self
        def __iter__(self): return iter([])
    np = MockNumpy
else:
    import numpy as np

def np_abs(x):
    if HAS_NUMPY:
        return np.abs(x)
    return [abs(i) for i in x]

@dataclass
class Signal:
    type: str  # 'BUY', 'SELL', or 'HOLD'
    price: float
    timestamp: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy:
    def __init__(self):
        self.data = None
        self.signals = []

    def init(self, data):
        self.data = data
        self.signals = []

    def next(self, bar_idx: int) -> Signal:
        raise NotImplementedError

    def get_signals(self) -> list[Signal]:
        return self.signals

    def get_params(self) -> dict:
        return {}

class RiskManagedStrategy(BaseStrategy):
    """
    A decorator-pattern strategy that wraps an underlying strategy 
    and enforces Stop-Loss and Take-Profit rules.
    """
    def __init__(self, underlying_strategy: BaseStrategy, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.05):
        super().__init__()
        self.strategy = underlying_strategy
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.active_position: Optional[Dict[str, Any]] = None 

    def init(self, data):
        self.data = data
        self.signals = []
        if hasattr(self, 'strategy'):
            self.strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        base_signal = self.strategy.next(bar_idx)
        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]

        # 1. Manage Existing Position
        if self.active_position:
            entry_price = self.active_position['price']
            direction = self.active_position['direction']
            # pnl_pct: positive if in profit, negative if in loss
            pnl_pct = (close_price - entry_price) / entry_price * direction

            # Stop Loss check
            if pnl_pct <= -self.stop_loss_pct:
                self.active_position = None
                return Signal('SELL' if direction == 1 else 'BUY', close_price, timestamp, 1.0, {'reason': 'stop_loss'})

            # Take Profit check
            if pnl_pct >= self.take_profit_pct:
                self.active_position = None
                return Signal('SELL' if direction == 1 else 'BUY', close_price, timestamp, 1.0, {'reason': 'take_profit'})

            # If the underlying strategy says HOLD or SELL while we are long, 
            # we might want to exit. But for this simple decorator, 
            # we let the underlying strategy signal the exit if it's not a risk event.
            if base_signal.type == 'HOLD':
                 return Signal('HOLD', close_price, timestamp, 0.0, {'reason': 'managing_position'})
            
            if base_signal.type == 'SELL' and direction == 1:
                self.active_position = None
                return base_signal

            if base_signal.type == 'BUY' and direction == -1:
                self.active_position = None
                return base_signal

        # 2. Entry Logic (only if not in a position)
        if base_signal.type in ['BUY', 'SELL'] and not self.active_position:
            direction = 1 if base_signal.type == 'BUY' else -1
            self.active_position = {
                'price': close_price,
                'direction': direction,
                'timestamp': timestamp
            }
            return base_signal

        return Signal('HOLD', close_price, timestamp, 0.0, {'reason': 'managing_position'})

class EnsembleStrategy(BaseStrategy):
    """
    A voting-based ensemble that takes multiple strategies and 
    emits a signal if a certain threshold of strategies agree.
    """
    def __init__(self, strategies: List[BaseStrategy], voting_threshold: float = 0.5):
        super().__init__()
        self.strategies = strategies
        self.voting_threshold = voting_threshold

    def init(self, data):
        self.data = data
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_strategies = len(self.strategies)
        for strategy in self.strategies:
            signal = strategy.next(bar_idx)
            votes[signal.type] += 1

        best_signal_type = 'HOLD'
        max_votes = 0
        for s_type in ['BUY', 'SELL']:
            # Check if the vote proportion meets threshold
            if total_strategies > 0 and (votes[s_type] / total_strategies) >= self.voting_threshold:
                if votes[s_type] > max_votes:
                    max_votes = votes[s_type]
                    best_signal_type = s_type

        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        confidence = max_votes / total_strategies if total_strategies > 0 else 0.0

        return Signal(
            type=best_signal_type,
            price=close_price,
            timestamp=timestamp,
            confidence=confidence,
            metadata={'ensemble_votes': votes}
        )

    def get_params(self) -> dict:
        params = {'voting_threshold': self.voting_threshold}
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params

class AdaptivePositionSizer(BaseStrategy):
    """
    Utility to calculate position size based on volatility (ATR) 
    and a fixed percentage of account equity.
    """
    def __init__(self, base_risk_pct: float = 0.01, atr_period: int = 14):
        super().__init__()
        self.base_risk_pct = base_risk_pct
        self.atr_period = atr_period

    def get_position_size(self, current_price: float, atr: float, account_equity: float) -> float:
        if atr <= 0:
            return 0.0
        # Amount of money to risk on this trade
        risk_amount = account_equity * self.base_risk_pct
        # Number of units = risk_amount / volatility_unit
        size = risk_amount / atr
        return size / current_price

class TrailingStopStrategy(BaseStrategy):
    """
    A template for a strategy that uses a trailing stop based on ATR.
    Note: This is a standalone strategy template, not a decorator.
    """
    def __init__(self, atr_multiplier: float = 2.0, atr_period: int = 14):
        super().__init__()
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self.highest_price = 0.0
        self.stop_price = 0.0
        self.active = False

    def next(self, bar_idx: int) -> Signal:
        if not self.active:
            if bar_idx <= 0:
                return Signal('HOLD', self.data['close'][bar_idx], self.data['timestamp'][bar_idx], 0.0)
            
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            # Calculate ATR manually if needed (simplified for template)
            tr = [0.0]*len(close)
            for i in range(1, len(close)):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            # Average TR
            start_idx = max(0, bar_idx - self.atr_period)
            relevant_tr = tr[start_idx:bar_idx+1]
            atr = sum(relevant_tr) / len(relevant_tr) if relevant_tr else 0.0
            
            if atr <= 0: 
                atr = close[bar_idx] * 0.01

            # Simple breakout entry
            if close[bar_idx] > high[bar_idx-1]:
                self.active = True
                self.highest_price = close[bar_idx]
                self.stop_price = close[bar_idx] - (self.atr_multiplier * atr)
                return Signal('BUY', close[bar_idx], self.data['timestamp'][bar_idx], 0.8)
            
            return Signal('HOLD', close[bar_idx], self.data['timestamp'][bar_idx], 0.0)

        # Manage existing position
        current_price = self.data['close'][bar_idx]
        
        # Update trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            tr = [0.0]*len(close)
            for i in range(1, len(close)):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            start_idx = max(0, bar_idx - self.atr_period)
            relevant_tr = tr[start_idx:bar_idx+1]
            atr = sum(relevant_tr) / len(relevant_tr) if relevant_tr else 0.0
            
            if atr > 0:
                self.stop_price = current_price - (self.atr_multiplier * atr)

        # Exit condition
        if current_price < self.stop_price:
            self.active = False
            return Signal('SELL', current_price, self.data['timestamp'][bar_idx], 1.0, {'reason': 'trailing_stop'})
        
        return Signal('HOLD', current_price, self.data['timestamp'][bar_idx], 0.0)

class MetaEnsembleStrategy(BaseStrategy):
    """
    A weighted ensemble strategy where each sub-strategy's signal 
    is multiplied by a weight and summed.
    """
    def __init__(self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None):
        super().__init__()
        self.strategies = strategies
        if weights is not None:
            if len(weights) != len(strategies):
                raise ValueError("Weights must match number of strategies")
            # Ensure weights are a numpy array for vector math
            if HAS_NUMPY:
                self.weights = np.array(weights)
            else:
                self.weights = weights
        else:
            self.weights = [1.0/len(strategies)] * len(strategies)

    def init(self, data):
        self.data = data
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        total_score = 0.0
        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]

        for i, strategy in enumerate(self.strategies):
            signal = strategy.next(bar_idx)
            score = 0.0
            if signal.type == 'BUY':
                score = signal.confidence
            elif signal.type == 'SELL':
                score = -signal.confidence
            
            # Handle weights safely
            try:
                w = self.weights[i]
            except (TypeError, IndexError):
                w = 1.0 / len(self.strategies)
                
            total_score += score * w

        # Decision threshold
        threshold = 0.05
        if total_score > threshold:
            best_type = 'BUY'
        elif total_score < -threshold:
            best_type = 'SELL'
        else:
            best_type = 'HOLD'

        return Signal(
            type=best_type,
            price=close_price,
            timestamp=timestamp,
            confidence=abs(total_score),
            metadata={'meta_score': total_score}
        )

    def get_params(self) -> dict:
        # Check for tolist method (numpy) or fallback
        try:
            w_list = self.weights.tolist() if hasattr(self.weights, 'tolist') else self.weights
        except:
            w_list = self.weights
            
        params = {'weights': w_list}
        for i, strategy in enumerate(self.strategies):
            params[f'strategy_{i}_params'] = strategy.get_params()
        return params
