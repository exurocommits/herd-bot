from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

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
        """
        Initialize the strategy with data.
        data: A dictionary or object containing 'timestamp', 'close', 'volume', etc.
        Expected data format: numpy arrays for values.
        """
        self.data = data
        self.signals = []

    def next(self, bar_idx: int) -> Signal:
        """
        Process a single bar.
        bar_idx: index of the current bar.
        """
        raise NotImplementedError


    def get_signals(self) -> list[Signal]:
        """Return all generated signals."""
        return self.signals

    def get_params(self) -> dict:
        """Return strategy parameters."""
        return {}

class RiskManagedStrategy(BaseStrategy):
    """
    An extension of BaseStrategy that adds built-in risk management
    logic to any underlying signal generator.
    """
    def __init__(self, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.05, max_position_size: float = 1.0):
        super().__init__()
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_size = max_position_size
        
        self.active_position = None # Stores entry price and direction
        self.last_signal = None

    def apply_risk_management(self, bar_idx: int, current_signal: Signal) -> Signal:
        """
        Wraps a raw signal with stop-loss and take-profit logic.
        """
        if self.data is None:
            return current_signal

        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]

        # 1. Check exit conditions if in a position
        if self.active_position:
            entry_price = self.active_position['price']
            direction = self.active_position['direction'] # 1 for Long, -1 for Short
            
            # Calculate PnL
            pnl_pct = (close_price - entry_price) / entry_price * direction

            # Stop Loss
            if pnl_pct <= -self.stop_loss_pct:
                self.active_position = None
                return Signal('SELL' if direction == 1 else 'BUY', close_price, timestamp, 1.0, {'reason': 'stop_loss'})

            # Take Profit
            if pnl_pct >= self.take_profit_pct:
                self.active_position = None
                return Signal('SELL' if direction == 1 else 'BUY', close_price, timestamp, 1.0, {'reason': 'take_profit'})
            
            # If we are already managing a position, don't let new signals override it unless it's a close
            if current_signal.type == 'HOLD':
                return Signal('HOLD', close_price, timestamp, 0.0, {'reason': 'managing_position'})

        # 2. Handle new entry signals
        if current_signal.type in ['BUY', 'SELL'] and not self.active_position:
            direction = 1 if current_signal.type == 'BUY' else -1
            self.active_position = {
                'price': close_price,
                'direction': direction,
                'timestamp': timestamp
            }
            return current_signal

        return current_signal

class EnsembleStrategy(BaseStrategy):
    """
    An ensemble strategy that combines multiple strategies by averaging their signals.
    Useful for reducing variance and improving robustness.
    """
    def __init__(self, strategies: list[BaseStrategy], voting_threshold: float = 0.5):
        super().__init__()
        self.strategies = strategies
        self.voting_threshold = voting_threshold

    def init(self, data):
        for strategy in self.strategies:
            strategy.init(data)

    def next(self, bar_idx: int) -> Signal:
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_strategies = len(self.strategies)

        for strategy in self.strategies:
            signal = strategy.next(bar_idx)
            votes[signal.type] += 1

        # Determine the winner
        best_signal_type = 'HOLD'
        max_votes = 0
        
        # We only care about BUY/SELL for the ensemble threshold
        for s_type in ['BUY', 'SELL']:
            if votes[s_type] / total_strategies >= self.voting_threshold:
                if votes[s_type] > max_votes:
                    max_votes = votes[s_type]
                    best_signal_type = s_type

        # Construct a synthetic signal
        # We use the close price and current timestamp
        close_price = self.data['close'][bar_idx]
        timestamp = self.data['timestamp'][bar_idx]
        
        # Confidence is represented by how many strategies agreed
        confidence = max_votes / total_strategies if max_votes > 0 else 0.0

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
