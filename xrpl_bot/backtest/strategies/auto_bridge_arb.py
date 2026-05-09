import numpy as np

class AutoBridgeArbStrategy:
    """
    XRPL auto-bridging strategy.
    Matches XRP-paired orders through XRP to exploit price discrepancies.
    
    Simulates: direct pair price (A->B) vs routed price (A->XRP->B).
    Route price = price_a_xrp * price_xrp_b.
    
    Logic:
    - If direct > routed * (1 + 0.003) -> BUY via route.
    - If direct < routed * (1 - 0.003) -> BUY direct.
    - Confidence: |direct - routed| / (0.003 * direct).
    """
    def __init__(self, data):
        """
        Initialize with data.
        data should be a dict or DataFrame containing:
        - 'direct_price': price of A/B
        - 'price_a_xrp': price of A/XRP
        - 'price_xrp_b': price of XRP/B
        """
        self.data = data
        self.index = 0
        self.signals = []

    def next(self, bar):
        """
        Processes the next bar of data.
        bar: dict containing 'direct_price', 'price_a_xrp', 'price_xrp_b'
        """
        direct = bar['direct_price']
        a_xrp = bar['price_a_xrp']
        xrp_b = bar['price_xrp_b']
        
        routed = a_xrp * xrp_b
        
        signal = {"type": "HOLD", "price": direct, "confidence": 0.0, "metadata": {}}
        
        # Arbitrage check
        # Case 1: Direct is too expensive compared to route
        if direct > routed * (1 + 0.003):
            # We want the cheaper one. If we are buying B with A, and route is cheaper, 
            # it implies the direct path is overpriced. 
            # The prompt says "BUY via route". 
            confidence = abs(direct - routed) / (0.003 * direct)
            signal = {
                "type": "BUY", 
                "price": routed, 
                "confidence": float(confidence), 
                "metadata": {"reason": "route_cheaper", "direct": direct, "routed": routed}
            }
        
        # Case 2: Direct is too cheap compared to route
        elif direct < routed * (1 - 0.003):
            confidence = abs(direct - routed) / (0.003 * direct)
            signal = {
                "type": "BUY", 
                "price": direct, 
                "confidence": float(confidence), 
                "metadata": {"reason": "direct_cheaper", "direct": direct, "routed": routed}
            }
            
        self.signals.append(signal)
        self.index += 1
        return signal

    def get_signals(self):
        """Returns all generated signals."""
        return self.signals

    def get_params(self):
        """Returns strategy parameters."""
        return {"threshold": 0.003}
