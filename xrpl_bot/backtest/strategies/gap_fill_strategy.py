import numpy as np

class GapFillStrategy:
    """
    Gap = open[0] - close[-1] > 1*ATR. 
    Gap up = SELL (expect fill). 
    Gap down = BUY (expect fill). 
    Target: previous close (gap fill). 
    Stop: 0.5*ATR against. 
    Timeout: 10 bars. 
    Confidence: gap size / ATR.
    """
    def __init__(self, data):
        self.data = data
        self.atr_period = 14
        self.position = None  # 'LONG', 'SHORT', or None
        self.entry_price = 0.0
        self.target_price = 0.0
        self.stop_loss = 0.0
        self.timeout = 10
        self.bars_since_entry = 0
        self.signals = []
        self.idx = 0
        self.atr = self._compute_atr()

    def _compute_atr(self):
        high, low, close = self.data[:, 1], self.data[:, 2], self.data[:, 3]
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        atr = np.zeros_like(close)
        for i in range(self.atr_period, len(close)):
            atr[i] = np.mean(tr[i-self.atr_period:i])
        return atr

    def next(self, bar):
        idx = self.idx
        if idx < self.atr_period + 1:
            self.idx += 1
            return {"type": "HOLD", "price": bar[3], "confidence": 0.0, "metadata": {}}

        c_open, c_high, c_low, c_close, c_atr = bar[0], bar[1], bar[2], bar[3], self.atr[idx]
        sig = {"type": "HOLD", "price": c_close, "confidence": 0.0, "metadata": {}}

        # Exit logic for active position
        if self.position:
            self.bars_since_entry += 1
            
            # 1. Check Target/Stop
            if self.position == 'LONG':
                if c_high >= self.target_price:
                    self.signals.append({"type": "SELL", "price": self.target_price, "confidence": 1.0, "metadata": {"reason": "target_hit"}})
                    self.position = None
                elif c_low <= self.stop_loss:
                    self.signals.append({"type": "SELL", "price": self.stop_loss, "confidence": 1.0, "metadata": {"reason": "stop_hit"}})
                    self.position = None
                elif self.bars_since_entry >= self.timeout:
                    self.signals.append({"type": "SELL", "price": c_close, "confidence": 1.0, "metadata": {"reason": "timeout"}})
                    self.position = None
            
            elif self.position == 'SHORT':
                if c_low <= self.target_price:
                    self.signals.append({"type": "BUY", "price": self.target_price, "confidence": 1.0, "metadata": {"reason": "target_hit"}})
                    self.position = None
                elif c_high >= self.stop_loss:
                    self.signals.append({"type": "BUY", "price": self.stop_loss, "confidence": 1.0, "metadata": {"reason": "stop_hit"}})
                    self.position = None
                elif self.bars_since_entry >= self.timeout:
                    self.signals.append({"type": "BUY", "price": c_close, "confidence": 1.0, "metadata": {"reason": "timeout"}})
                    self.position = None
            
            if self.position is None:
                self.idx += 1
                return sig

        # Entry logic
        if self.position is None:
            prev_close = self.data[idx-1, 3]
            gap = c_open - prev_close
            
            if abs(gap) > c_atr:
                conf = abs(gap) / c_atr
                if gap < 0: # Gap down -> BUY
                    self.position = 'LONG'
                    self.entry_price = c_open
                    self.target_price = prev_close
                    self.stop_loss = c_open - 0.5 * c_atr
                    self.bars_since_entry = 0
                    sig = {"type": "BUY", "price": c_open, "confidence": conf, "metadata": {"gap": gap}}
                    self.signals.append(sig)
                elif gap > 0: # Gap up -> SELL
                    self.position = 'SHORT'
                    self.entry_price = c_open
                    self.target_price = prev_close
                    self.stop_loss = c_open + 0.5 * c_atr
                    self.bars_since_entry = 0
                    sig = {"type": "SELL", "price": c_open, "confidence": conf, "metadata": {"gap": gap}}
                    self.signals.append(sig)

        self.idx += 1
        return sig

    def get_signals(self): return self.signals
    def get_params(self): return {"atr_period": self.atr_period, "timeout": self.timeout}
