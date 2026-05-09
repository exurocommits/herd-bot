import numpy as np

class AMMSingleHopArb:
    """
    Constant product AMM arbitrage strategy (x*y=k).
    Identifies price discrepancies between an AMM pool and a CEX.
    """

    def __init__(self, data):
        """
        Initialize the strategy.
        :param data: DataFrame or dict containing 'pool_reserve_a', 'pool_reserve_b', 'cex_price'
        """
        self.data = data
        self.min_profit = 0.005
        self.current_index = 0
        self.history = []

    def next(self, bar):
        """
        Processes the next bar of data.
        :param bar: dict containing 'pool_reserve_a', 'pool_reserve_b', 'cex_price'
        :return: dict containing signal information
        """
        self.current_index += 1
        
        reserve_a = bar['pool_reserve_a']
        reserve_b = bar['pool_reserve_b']
        cex_price = bar['cex_price']

        # AMM price (price of A in terms of B) = reserve_b / reserve_a
        amm_price = reserve_b / reserve_a
        
        signal = {"type": "HOLD", "price": float(cex_price), "confidence": 0.0, "metadata": {}}

        # Check for BUY opportunity (AMM is cheap)
        # amm_price < cex_price * (1 - min_profit)
        if amm_price < cex_price * (1 - self.min_profit):
            profit_margin = (cex_price / amm_price) - 1
            confidence = min(profit_margin / self.min_profit, 1.0)
            
            signal = {
                "type": "BUY",
                "price": float(amm_price),
                "confidence": float(confidence),
                "metadata": {
                    "amm_price": float(amm_price),
                    "cex_price": float(cex_price),
                    "profit_margin": float(profit_margin)
                }
            }

        # Check for SELL opportunity (AMM is expensive)
        # amm_price > cex_price * (1 + min_profit)
        elif amm_price > cex_price * (1 + self.min_profit):
            profit_margin = (amm_price / cex_price) - 1
            confidence = min(profit_margin / self.min_profit, 1.0)
            
            signal = {
                "type": "SELL",
                "price": float(amm_price),
                "confidence": float(confidence),
                "metadata": {
                    "amm_price": float(amm_price),
                    "cex_price": float(cex_price),
                    "profit_margin": float(profit_margin)
                }
            }

        self.history.append(signal)
        return signal

    def get_signals(self):
        """
        Returns the list of signals generated so far.
        """
        return self.history

    def get_params(self):
        """
        Returns the strategy parameters.
        """
        return {
            "min_profit": self.min_profit
        }
