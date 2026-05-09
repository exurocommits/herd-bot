from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
from .base_strategy import BaseStrategy, Signal

class AMMPoolArbitrage(BaseStrategy):
    """
    Strategy 15: AMM Pool Arbitrage
    Simulates arbitrage between a Constant Product AMM pool and a CEX.
    """
    def __init__(self, min_profit_margin: float = 0.005):
        super().__init__()
        self.min_profit_margin = min_profit_margin

    def init(self, data: Dict[str, Any]):
        """
        data: dict containing:
            'reserve_a': current reserve of asset A in pool
            'reserve_b': current reserve of asset B in pool
            'cex_price': price of asset A in terms of asset B on CEX
            'timestamp': current timestamp
        """
        self.data = data
        self.signals = []
        # In a real backtest, these would be arrays/series
        self.reserves_a = data['reserve_a']
        self.reserves_b = data['reserve_b']
        self.cex_prices = data['cex_price']
        self.timestamps = data['timestamp']

    def next(self, bar_idx: int) -> Signal:
        res_a = self.reserves_a[bar_idx]
        res_b = self.reserves_b[bar_idx]
        cex_price = self.cex_prices[bar_idx]
        ts = self.timestamps[bar_idx]
        
        if res_a <= 0 or res_b <= 0:
            return Signal('HOLD', cex_price, ts, 0.0)

        # Constant Product: x * y = k
        k = res_a * res_b
        
        # AMM effective price (price of A in terms of B)
        # As we approach 0 reserve_a, price goes to infinity.
        # The current spot price is res_b / res_a
        amm_price = res_b / res_a
        
        signal_type = 'HOLD'
        confidence = 0.0
        
        # Check for arbitrage opportunities
        # Case 1: AMM price > CEX price -> Asset A is expensive on AMM.
        # We should SELL A on AMM and BUY A on CEX.
        # Wait, the prompt says: "If AMM price - CEX price > min_profit_margin -> BUY on AMM, SELL on CEX"
        # Let's follow the prompt's specific logic even if the direction seems flipped 
        # (depending on whether price is A/B or B/A).
        # Prompt: "If AMM price - CEX price > min_profit_margin (0.5%) -> BUY on AMM, SELL on CEX"
        
        if (amm_price - cex_price) > self.min_profit_margin:
            signal_type = 'BUY' # "BUY on AMM"
            # Note: This logic is simplified. In real life, we'd need to calculate slippage.
            # new_price = k / (reserve - trade_size) - price
            # We'll use a proxy for confidence based on the margin.
            confidence = (amm_price - cex_price) / self.min_profit_margin
            
        elif (cex_price - amm_price) > self.min_profit_margin:
            signal_type = 'SELL' # "SELL on AMM"
            confidence = (cex_price - amm_price) / self.min_profit_margin

        # Account for slippage (simplified approximation for the signal metadata)
        # We'll assume a small trade size to estimate slippage
        trade_size = 1.0 
        if res_a > trade_size:
            slippage_price = k / (res_a - trade_size) # This is a rough way to show the impact
            # (Actually slippage affects the effective execution price)
        
        sig = Signal(
            type=signal_type,
            price=cex_price,
            timestamp=ts,
            confidence=min(float(confidence), 1.0),
            metadata={
                'amm_price': float(amm_price),
                'cex_price': float(cex_price),
                'margin': float(abs(amm_price - cex_price))
            }
        )
        self.signals.append(sig)
        return sig

    def get_signals(self) -> list[Signal]:
        return [s for s in self.signals if s.type != 'HOLD']

    def get_params(self) -> dict:
        return {'min_profit_margin': self.min_profit_margin}
