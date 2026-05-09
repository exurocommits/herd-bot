import numpy as np

class VortexIndicator:
    """
    Vortex Indicator (VI+ and VI- over period 14).
    VI+ = VM+ / TR_sum, VI- = VM- / TR_sum.
    VM+ = |high - prev_low|, VM- = |low - prev_high|.
    BUY when VI+ crosses above VI-.
    SELL when VI- crosses above VI+.
    Confidence: |VI+ - VI-| / (VI+ + VI-).
    """
    def __init__(self, data):
        """
        :param data: Dictionary containing 'high', 'low', 'close' as numpy arrays.
        """
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.period = 14
        self.current_index = 0
        self.position = None

        # Pre-calculate components
        self.vi_plus, self.vi_minus = self._calculate_vortex()

    def _calculate_vortex(self):
        n = len(self.close)
        vm_plus = np.zeros(n)
        vm_minus = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            vm_plus[i] = abs(self.high[i] - self.low[i-1])
            vm_minus[i] = abs(self.low[i] - self.high[i-1])
            
            # True Range calculation
            tr_high = max(self.high[i], self.close[i-1])
            tr_low = min(self.low[i], self.close[i-1])
            tr[i] = max(self.high[i] - self.low[i], 
                        abs(self.high[i] - self.close[i-1]), 
                        abs(self.low[i] - self.close[i-1]))

        vi_plus = np.zeros(n)
        vi_minus = np.zeros(n)

        for i in range(self.period, n):
            tr_sum = np.sum(tr[i - self.period + 1 : i + 1])
            if tr_sum > 0:
                vi_plus[i] = np.sum(vm_plus[i - self.period + 1 : i + 1]) / tr_sum
                vi_minus[i] = np.sum(vm_minus[i - self.period + 1 : i + 1]) / tr_sum

        return vi_plus, vi_minus

    def next(self, bar):
        idx = self.current_index
        self.current_index += 1
        
        current_close = bar['close']
        if idx < self.period + 1:
            return {"type": "HOLD", "price": current_close, "confidence": 0.0, "metadata": {}}

        v_plus = self.vi_plus[idx]
        v_minus = self.vi_minus[idx]
        prev_v_plus = self.vi_plus[idx-1]
        prev_v_minus = self.vi_minus[idx-1]

        signal_type = "HOLD"
        confidence = 0.0
        
        # Confidence: |VI+ - VI-| / (VI+ + VI-)
        denom = (v_plus + v_minus)
        confidence = abs(v_plus - v_minus) / denom if denom > 0 else 0.0

        if self.position is None:
            # BUY when VI+ crosses above VI-
            if v_plus > v_minus and prev_v_plus <= prev_v_minus:
                self.position = "BUY"
                signal_type = "BUY"
        
        elif self.position == "BUY":
            # SELL when VI- crosses above VI+
            if v_minus > v_plus and prev_v_minus <= prev_v_plus:
                self.position = None
                signal_type = "SELL"

        return {
            "type": signal_type,
            "price": float(current_close),
            "confidence": float(confidence),
            "metadata": {"vi_plus": float(v_plus), "vi_minus": float(v_minus)}
        }

    def get_signals(self):
        return []

    def get_params(self):
        return {"period": self.period}
