import numpy as np

class BTCDominanceRotation:
    \"\"\"
    BTC Dominance Rotation Strategy.
    
    Takes BTC price and alt price. BTC dominance proxy = btc_market_proxy / (btc_proxy + alt_proxy).
    Uses 50-bar MA of dominance to identify alt seasons.
    
    BUY alt when dominance falls below MA.
    BUY BTC when dominance rises above MA.
    \"\"\"
    def __init__(self, data):
        \"\"\"
        :param data: dict containing 'btc_price' and 'alt_price' as numpy arrays.
        \"\"\"
        self.btc_prices = data['btc_price']
        self.alt_prices = data['alt_price']
        self.constant = 1.0  # Constant for proxy calculation
        self.window = 50
        self.current_index = 0
        self.dominance_history = []
        
    def next(self, bar):
        \"\"\"
        Processes the next bar.
        :param bar: dict containing 'btc_price' and 'alt_price' for the current bar.
        :return: dict signal.
        \"\"\"
        self.current_index += 1
        btc_p = bar['btc_price']
        alt_p = bar['alt_price']
        
        btc_proxy = btc_p * self.constant
        alt_proxy = alt_p * self.constant
        dominance = btc_proxy / (btc_proxy + alt_proxy)
        
        self.dominance_history.append(dominance)
        
        if len(self.dominance_history) < self.window:
            return {"type": "HOLD", "price": btc_p, "confidence": 0.0, "metadata": {}}
            
        dom_arr = np.array(self.dominance_history[-self.window:])
        ma = np.mean(dom_arr)
        std = np.std(dom_arr)
        
        confidence = abs(dominance - ma) / std if std > 0 else 0.0
        
        signal_type = "HOLD"
        if dominance < ma:
            signal_type = "BUY_ALT" # Logic implies alt season
        elif dominance > ma:
            signal_type = "BUY_BTC"
            
        # Mapping to the requested signal types
        # The requirement says BUY alt or BUY BTC. 
        # Since the return type is single 'type', we'll use descriptive types or handle via metadata.
        # To strictly follow "BUY"/"SELL"/"HOLD", let's adjust:
        # If we are in alt season, we want to buy alts. If we are in btc season, buy btc.
        # This strategy is a bit ambiguous on whether 'type' refers to the asset or the action.
        # I will use BUY/SELL/HOLD and specify asset in metadata.
        
        # Actually, let's interpret: BUY means "enter the position favored by the signal"
        # Let's use: BUY (for the signal's preferred asset), SELL (to exit), HOLD.
        # Or better: The requirement says 'BUY'/'SELL'/'HOLD'.
        # If dominance < MA -> Alt Season -> We want to hold Alts.
        # If dominance > MA -> BTC Season -> We want to hold BTC.
        
        # Let's refine to:
        # BUY if dominance < MA (buying Alts)
        # SELL if dominance > MA (selling Alts, buying BTC) - wait, the requirement says BUY BTC when dominance rises.
        
        # Let's follow the prompt literally:
        # BUY alt when dominance < MA.
        # BUY BTC when dominance > MA.
        
        # To fit "BUY"/"SELL"/"HOLD", I'll define:
        # If dominance < MA -> Signal: BUY (Asset: ALT)
        # If dominance > MA -> Signal: BUY (Asset: BTC)
        # This is tricky for a single signal dict. 
        # I'll provide 'type': 'BUY' and 'asset' in metadata.
        
        # Re-reading: "BUY alt when dominance falls below MA (alt season). BUY BTC when dominance rises above MA."
        # This sounds like a switching strategy.
        
        # Let's provide:
        # if dominance < MA and previous_state != 'ALT': BUY (alt)
        # if dominance > MA and previous_state != 'BTC': BUY (btc)
        
        # For the sake of the prompt's specific format:
        return {
            "type": "BUY" if signal_type != "HOLD" else "HOLD",
            "price": btc_p,
            "confidence": float(confidence),
            "metadata": {
                "signal_logic": signal_type,
                "dominance": float(dominance),
                "ma": float(ma)
            }
        }

    def get_signals(self):
        \"\"\"Placeholder for batch processing if needed.\"\"\"
        return []

    def get_params(self):
        return {"constant": self.constant, "window": self.window}
