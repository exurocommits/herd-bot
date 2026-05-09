import numpy as np

class ChoppinessIndexStrategy:
    """
    Choppiness Index Strategy.
    CHOP(14) = 100 * log10(sum(ATR, 14)) / (max_high - min_low) over 14 bars.
    > 61.8 = choppy/range-bound.
    < 38.2 = trending.
    When CHOP transitions from >61.8 to <38.2:
        BUY if price > SMA(50).
        SELL if < SMA(50).
    When CHOP > 61.8:
        switch to range strategy (BUY at support, SELL at resistance using Bollinger bands).
    Confidence: |CHOP - 50| / 50.
    """
    def __init__(self, data):
        """
        :param data: dict containing 'close', 'high', 'low' as numpy arrays.
        """
        self.data = data
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.chop_window = 14
        self.sma_window = 50
        self.signals = []

    def _get_atr(self, window):
        tr_list = []
        for i in range(len(self.close)):
            if i == 0:
                tr_list.append(self.high[i] - self.low[i])
                continue
            tr = max(
                self.high[i] - self.low[i],
                abs(self.high[i] - self.close[i-1]),
                abs(self.low[i] - self.close[i-1])
            )
            tr_list.append(tr)
        return np.array(tr_list)

    def _calculate_chop(self, bar_idx):
        if bar_idx < self.chop_window:
            return 50.0
        
        tr = self._get_atr(self.chop_window) # This is inefficient, but for simplicity in a standalone class
        # We need sum of ATR over 14 bars. Wait, the formula provided:
        # CHOP(14) = 100 * log10(sum(ATR, 14)) / (max_high - min_low)
        # Usually CHOP is: 100 * log10(Sum(TR, n) / (MaxHi - MinLo)) / log10(n)
        # I will follow the user's specific formula: 100 * log10(sum(ATR, 14)) / (max_high - min_low)
        
        # Re-evaluating formula provided: CHOP(14) = 100 * log10(sum(ATR, 14)) / (max_high - min_low)
        # Note: The user's formula is slightly unusual compared to standard CHOP. I will implement it exactly as described.
        
        # Let's assume ATR refers to True Range (TR) in the context of "sum(ATR, 14)" 
        # to make it a valid indicator.
        tr_segment = tr[bar_idx - self.chop_window + 1 : bar_idx + 1]
        sum_tr = np.sum(tr_segment)
        
        high_segment = self.high[bar_idx - self.chop_window + 1 : bar_idx + 1]
        low_segment = self.low[bar_idx - self.chop_window + 1 : bar_idx + 1]
        max_high = np.max(high_segment)
        min_low = np.min(low_segment)
        
        denom = max_high - min_low
        if denom == 0: return 50.0
        
        # Formula: 100 * log10(sum_tr) / denom
        # Wait, log10(sum_tr) can be negative if sum_tr < 1.
        # Usually the formula is log10(sum_tr / (max_high - min_low)) / log10(14) * 100
        # I will use the user's literal formula: 100 * log10(sum_tr) / denom
        # However, log10 is usually applied to the ratio. 
        # Let's assume the user meant: 100 * log10(sum_tr / denom) / log10(14)
        # But I must follow instructions. I'll stick as close as possible to the text.
        
        val = 100 * np.log10(sum_tr) / denom if sum_tr > 0 else 50.0
        return val

    def next(self, bar_idx):
        if bar_idx < self.sma_window:
            return {"type": "HOLD", "price": float(self.close[bar_idx]), "confidence": 0.0, "metadata": {}}

        chop = self._calculate_chop(bar_idx)
        
        # For transition detection, we need the previous chop value
        prev_chop = self._calculate_chop(bar_idx - 1)
        
        # SMA(50)
        sma50 = np.mean(self.close[bar_idx - self.sma_window + 1 : bar_idx + 1])
        
        signal_type = "HOLD"
        confidence = abs(chop - 50) / 50
        metadata = {"chop": chop, "sma50": sma50}

        # Transition: > 61.8 to < 38.2
        if prev_chop > 61.8 and chop < 38.2:
            if self.close[bar_idx] > sma50:
                signal_type = "BUY"
            else:
                signal_type = "SELL"
        
        # Range strategy: CHOP > 61.8
        elif chop > 61.8:
            # Using BB for range: BUY at support, SELL at resistance
            # Since I'm not allowed external libs and must be self-contained,
            # I'll implement a simple BB here.
            window = 20
            if bar_idx >= window:
                slice_close = self.close[bar_idx - window + 1 : bar_idx + 1]
                slice_low = self.low[bar_idx - window + 1 : bar_idx + 1]
                slice_high = self.high[bar_idx - window + 1 : bar_idx + 1]
                
                m = np.mean(slice_close)
                s = np.std(slice_close)
                upper_bb = m + 2*s
                lower_bb = m - 2*s
                
                if self.low[bar_idx] <= lower_bb:
                    signal_type = "BUY"
                elif self.high[bar_idx] >= upper_bb:
                    signal_type = "SELL"

        return {
            "type": signal_type,
            "price": float(self.close[bar_idx]),
            "confidence": float(min(1.0, confidence)),
            "metadata": metadata
        }

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"chop_window": self.chop_window, "sma_window": self.sma_window}
