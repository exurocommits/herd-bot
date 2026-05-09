import numpy as np

class ChandeMomentumOscillator:
    """
    Chande Momentum Oscillator (CMO) Strategy.
    CMO(14) = (sum_up - sum_down) / (sum_up + sum_down) * 100.
    BUY when CMO crosses above 0 from below -50.
    SELL when CMO crosses below 0 from above +50.
    Confidence: |CMO| / 100.
    """

    def __init__(self, data):
        """
        :param data: A numpy array of closing prices.
        """
        self.prices = data
        self.period = 14
        self.current_index = 0
        self._cmo = self._calculate_cmo(self.prices)

    def _calculate_cmo(self, data):
        n = len(data)
        cmo = np.full(n, np.nan)
        
        if n <= self.period:
            return cmo

        diffs = np.diff(data)
        
        for i in range(self.period, n):
            # diffs[i-1] is data[i] - data[i-1]
            # We need the last 'period' differences
            # The slice is from i-period to i
            window_diffs = diffs[i - self.period : i]
            
            up = np.sum(window_diffs[window_diffs > 0])
            down = np.sum(np.abs(window_diffs[window_diffs < 0]))
            
            if (up + down) == 0:
                cmo[i] = 0
            else:
                cmo[i] = ((up - down) / (up + down)) * 100
                
        return cmo

    def next(self, bar):
        idx = self.current_index
        if idx >= len(self.prices):
            return {"type": "HOLD", "price": bar, "confidence": 0.0, "metadata": {}}

        price = bar
        signal = {"type": "HOLD", "price": price, "confidence": 0.0, "metadata": {}}

        if idx > 0 and not np.isnan(self._cmo[idx]) and not np.isnan(self._cmo[idx-1]):
            cmo_now = self._cmo[idx]
            cmo_prev = self._cmo[idx-1]

            # BUY when CMO crosses above 0 from below -50
            if cmo_prev < -50 and cmo_now > 0:
                # Actually "crosses above 0 from below -50" implies a range.
                # But usually, it means it was below -50 and is now > 0.
                # Let's refine: if it was < -50 and is now > 0.
                # Or does it mean it crosses 0, provided it was recently below -50?
                # Let's go with: cmo_prev < 0 and cmo_now > 0 and some condition about -50.
                # The prompt is slightly ambiguous. Let's assume: 
                # (cmo_prev < 0 and cmo_now > 0) AND (the minimum in the window was < -50)
                # Simpler: cmo_prev < 0 and cmo_now > 0, and we check if it was "deep"
                # Let's stick to: cmo_prev < -50 and cmo_now > 0 is too restrictive.
                # Let's assume it means: it crosses 0, and the crossover is triggered if it was below -50.
                # Actually, "crosses above 0 from below -50" is a specific event.
                # Let's check if cmo_prev <= 0 and cmo_now > 0 and we track if it hit -50.
                # To keep it simple and strictly follow:
                pass 

            # Let's try a simpler interpretation:
            # BUY: cmo_prev <= 0 and cmo_now > 0 AND cmo_prev was part of a dip below -50.
            # Since we don't have full history easily, let's use:
            # BUY: cmo_prev < 0 and cmo_now > 0 and we check if cmo_prev was < -50.
            # Actually, the most literal is: cmo_prev <= 0 and cmo_now > 0 AND the value was < -50 recently.
            # Let's just do:
            if cmo_prev <= 0 and cmo_now > 0:
                # Check if it was below -50 in the last few bars? 
                # Let's just check if cmo_prev < 0 and it's a crossover.
                # Given the wording, let's assume it means the crossover happens at 0,
                # but it's only valid if it came from a -50 zone.
                # Since we don't have the full "dip" state, we'll use the previous value.
                if cmo_prev <= -50 or (idx > 1 and self._cmo[idx-1] <= -50): # Very loose
                   pass

            # Let's use a more standard interpretation of such signals:
            # BUY: crossover 0, having been below -50.
            # Because I can't easily know "was below -50" without state, 
            # I'll check if cmo_prev is between -50 and 0 and cmo_now > 0? No.
            # Let's try: cmo_prev <= 0 and cmo_now > 0 AND we'll assume the condition is met if cmo_prev is low.
            # Let's just implement the literal crossing:
            if cmo_prev <= 0 and cmo_now > 0 and cmo_prev < 0:
                # If the prompt implies the crossover itself is the trigger:
                # We'll check if the current crossover is "from below -50" 
                # by checking if the dip occurred.
                # For simplicity in this task:
                if cmo_prev < 0: # This is a crossover
                    # To satisfy "from below -50", let's check if it was < -50 in the last 10 bars.
                    # But we'll just do a simple:
                    if cmo_prev <= 0 and cmo_now > 0:
                         # We will treat any 0-crossing as valid if we assume the context.
                         # But to be safe, let's just look for the crossover.
                         pass

            # RE-READ: "BUY when CMO crosses above 0 from below -50"
            # This is a specific condition. Let's use a flag or just check the previous value.
            # If cmo_prev <= 0 and cmo_now > 0, and we assume it came from -50.
            # Let's just implement:
            if cmo_prev <= 0 and cmo_now > 0:
                # We'll check if it was below -50 in the last 'period' bars.
                # This is a good way to implement "from below -50".
                if np.any(self._cmo[max(0, idx-self.period):idx] <= -50):
                    conf = min(abs(cmo_now) / 100.0, 1.0)
                    signal = {"type": "BUY", "price": price, "confidence": conf, "metadata": {"cmo": cmo_now}}
            
            # SELL when CMO crosses below 0 from above +50
            elif cmo_prev >= 0 and cmo_now < 0:
                if np.any(self._cmo[max(0, idx-self.period):idx] >= 50):
                    conf = min(abs(cmo_now) / 100.0, 1.0)
                    signal = {"type": "SELL", "price": price, "confidence": conf, "metadata": {"cmo": cmo_now}}

        self.current_index += 1
        return signal

    def get_signals(self):
        return []

    def get_params(self):
        return {"period": self.period}
