import numpy as np

class AMMMultiHopArb:
    """
    Triangular arbitrage strategy through 3 AMM pools: A->B, B->C, C->A.
    """

    def __init__(self, data):
        """
        Initialize the strategy.
        :param data: Historical data containing pool reserves for 3 pairs
        """
        self.data = data
        self.min_profit = 0.003
        self.current_index = 0
        self.history = []

    def next(self, bar):
        """
        Processes the next bar of data.
        :param bar: dict containing reserves for pairs AB, BC, and CA
        :return: dict containing signal information
        """
        self.current_index += 1
        
        # bar should contain reserves for:
        # 'res_a_b' (pair A/B), 'res_b_c' (pair B/C), 'res_c_a' (pair C/A)
        # Each is a tuple/dict (reserve_a, reserve_b)
        
        r_ab = bar['res_a_b']
        r_bc = bar['res_b_c']
        r_ca = bar['res_c_a']

        # Start with 1000 units of A
        amount_a = 1000.0

        # 1. A -> B
        # x * y = k => reserve_a * reserve_b = k
        # amount_b_out = (reserve_b * amount_a) / (reserve_a + amount_a)
        b_out = (r_ab[1] * amount_a) / (r_ab[0] + amount_a)

        # 2. B -> C
        # amount_c_out = (reserve_c * amount_b) / (reserve_b + amount_b)
        c_out = (r_bc[1] * b_out) / (r_bc[0] + b_out)

        # 3. C -> A
        # amount_a_final = (reserve_a * amount_c) / (reserve_c + amount_c)
        a_final = (r_ca[1] * c_out) / (r_ca[0] + c_out)

        profit_ratio = a_final / amount_a
        signal = {"type": "HOLD", "price": 0.0, "confidence": 0.0, "metadata": {}}

        if profit_ratio > (1 + self.min_profit):
            profit_margin = profit_ratio - 1
            confidence = min(profit_margin / self.min_profit, 1.0)
            
            signal = {
                "type": "BUY",
                "price": 0.0, # Not strictly applicable for the whole route in this abstraction
                "confidence": float(confidence),
                "metadata": {
                    "a_final": float(a_final),
                    "profit_ratio": float(profit_ratio),
                    "profit_margin": float(profit_margin)
                }
            }
        
        self.history.append(signal)
        return signal

    def get_signals(self):
        return self.history

    def get_params(self):
        return {"min_profit": self.min_profit}
