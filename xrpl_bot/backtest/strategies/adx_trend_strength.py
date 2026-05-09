import numpy as np

class ADXTrendStrength:
    """
    ADX Trend Strength strategy.
    ADX(14). +DI and -DI(14).
    BUY when +DI > -DI AND ADX > 25 (strong trend).
    SELL when -DI > +DI AND ADX > 25.
    HOLD when ADX < 20 (no trend).
    Confidence: ADX value / 50.
    """
    def __init__(self, data):
        """
        Initialize with data.
        :param data: dict containing 'high', 'low', 'close' as numpy arrays.
        """
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.n = len(self.close)
        
        self.adx, self.plus_di, self.minus_di = self._calculate_adx(self.high, self.low, self.close, 14)
        self.signals = []
        self.current_idx = 0

    def _calculate_adx(self, high, low, close, period):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
                
        # Smoothed TR, +DM, -DM
        str_smoothed = np.zeros_like(tr)
        splus_dm_smoothed = np.zeros_like(plus_dm)
        sminus_dm_smoothed = np.zeros_like(minus_dm)
        
        # Initial ATR-like smoothing (using sum for first period)
        str_smoothed[period] = np.sum(tr[1:period+1])
        splus_dm_smoothed[period] = np.sum(plus_dm[1:period+1])
        sminus_dm_smoothed[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period + 1, len(high)):
            str_smoothed[i] = str_smoothed[i-1] - (str_smoothed[i-1]/period) + tr[i]
            splus_dm_smoothed[i] = splus_dm_smoothed[i-1] - (splus_dm_smoothed[i-1]/period) + plus_dm[i]
            sminus_dm_smoothed[i] = sminus_dm_smoothed[i-1] - (sminus_dm_smoothed[i-1]/period) + minus_dm[i]

        plus_di = 100 * (splus_dm_smoothed / str_smoothed)
        minus_di = 100 * (sminus_dm_smoothed / str_smoothed)
        
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        
        adx = np.zeros_like(dx)
        # Simple smoothing for ADX
        if len(dx) > period:
            adx[period*2] = np.mean(dx[period:period*2])
            for i in range(period*2 + 1, len(dx)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
                
        return adx, plus_di, minus_di

    def next(self, bar_idx):
        if bar_idx < 30 or bar_idx >= self.n:
            return None
            
        self.current_idx = bar_idx
        price = self.close[bar_idx]
        adx_val = self.adx[bar_idx]
        p_di = self.plus_di[bar_idx]
        m_di = self.minus_di[bar_idx]
        
        signal_type = "HOLD"
        confidence = 0.0
        
        if adx_val < 20:
            signal_type = "HOLD"
            confidence = 0.0
        elif p_di > m_di and adx_val > 25:
            signal_type = "BUY"
            confidence = min(adx_val / 50.0, 1.0)
        elif m_di > p_di and adx_val > 25:
            signal_type = "SELL"
            confidence = min(adx_val / 50.0, 1.0)
            
        signal = {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {
                "adx": float(adx_val),
                "plus_di": float(p_di),
                "minus_di": float(m_di)
            }
        }
        self.signals.append(signal)
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"period": 14}
