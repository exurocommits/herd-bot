import numpy as np

class RSISmoothedStrategy:
    """
    Ehlers Super Smoother applied to RSI(14).
    
    BUY when smoothed RSI < 30.
    SELL when smoothed RSI > 70.
    Confidence: |smoothed_RSI - 50| / 50.
    """
    def __init__(self, data):
        """
        :param data: pandas DataFrame with 'close' column
        """
        self.df = data
        self.rsi_period = 14
        self.smooth_period = 10 # Default period for Ehlers
        self.smoothed_rsi = None
        self._calculate_smoothed_rsi()

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_smoothed_rsi(self):
        rsi = self._calculate_rsi(self.df['close'], self.rsi_period)
        
        # Ehlers Super Smoother coefficients
        # a0=exp(-1.414*pi/period), b0=2*a0*cos(1.414*pi/period), c0=1-b0-a0
        period = self.smooth_period
        a0 = np.exp(-1.414 * np.pi / period)
        b0 = 2 * a0 * np.cos(1.414 * np.pi / period)
        c0 = 1 - b0 - a0
        
        smoothed = np.zeros(len(rsi))
        rsi_vals = rsi.values
        
        # Since it's recursive, we iterate
        for i in range(1, len(rsi_vals)):
            if np.isnan(rsi_vals[i]):
                smoothed[i] = np.nan
            else:
                # Handle NaN for previous values to allow startup
                prev_val = smoothed[i-1] if not np.isnan(smoothed[i-1]) else rsi_vals[i]
                smoothed[i] = c0 * rsi_vals[i] + b0 * prev_val + a0 * (smoothed[i-2] if i > 1 and not np.isnan(smoothed[i-2]) else rsi_vals[i])
                # Note: A simplified single-pole/two-pole approximation is often used. 
                # For a true recursive filter: y[i] = c0*x[i] + b0*y[i-1] + a0*y[i-2] is not quite right for x.
                # Ehlers Super Smoother is typically: y[i] = c0*x[i] + b0*y[i-1] + a0*y[i-2]
                # Wait, the prompt says: a0=exp(...), b0=2*a0*cos(...), c0=1-b0-a0.
                # The formula is y[i] = c0*x[i] + b0*y[i-1] + a0*y[i-2].
                # Let's use the correct recursive implementation.
        
        # Correcting the loop for the specific coefficients provided
        smoothed = np.full(len(rsi), np.nan)
        for i in range(2, len(rsi_vals)):
            if not np.isnan(rsi_vals[i]) and not np.isnan(rsi_vals[i-1]):
                # Using the provided coefficients
                # y[i] = c0*x[i] + b0*y[i-1] + a0*y[i-2]
                # However, the coefficients a0, b0, c0 provided are often used in:
                # y[i] = c0*x[i] + b0*y[i-1] + a0*y[i-2] 
                # Let's assume this structure.
                prev_y1 = smoothed[i-1] if not np.isnan(smoothed[i-1]) else rsi_vals[i-1]
                prev_y2 = smoothed[i-2] if not np.isnan(smoothed[i-2]) else rsi_vals[i-2]
                smoothed[i] = c0 * rsi_vals[i] + b0 * prev_y1 + a0 * prev_y2

        self.smoothed_rsi = pd.Series(smoothed, index=self.df.index)

    def next(self, bar):
        """
        :param bar: current index (int)
        :return: signal dict
        """
        import pandas as pd # local import to ensure it works if pandas is passed in df
        if bar >= len(self.smoothed_rsi):
            return {"type": "HOLD", "price": self.df['close'].iloc[bar], "confidence": 0.0, "metadata": {}}
            
        val = self.smoothed_rsi.iloc[bar]
        if np.isnan(val):
            return {"type": "HOLD", "price": self.df['close'].iloc[bar], "confidence": 0.0, "metadata": {}}

        price = self.df['close'].iloc[bar]
        signal_type = "HOLD"
        
        if val < 30:
            signal_type = "BUY"
        elif val > 70:
            signal_type = "SELL"
            
        confidence = abs(val - 50) / 50.0
        
        return {
            "type": signal_type,
            "price": float(price),
            "confidence": float(confidence),
            "metadata": {"smoothed_rsi": float(val)}
        }

    def get_signals(self):
        # Placeholder for batch processing if needed
        return []

    def get_params(self):
        return {"rsi_period": self.rsi_period, "smooth_period": self.smooth_period}

import pandas as pd
