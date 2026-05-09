import numpy as np

class PriceChannelBreakout:
    """
    20-period high/low channel strategy.
    BUY when close > highest high of last 20 bars.
    SELL when close < lowest low.
    Exit when price touches channel midline.
    Confidence: breakout distance / channel width.
    """
    def __init__(self, data):
        """
        :param data: Dictionary containing 'close', 'high', 'low' as numpy arrays.
        """
        self.close = data['close']
        self.high = data['high']
        self.low = data['low']
        self.period = 20
        self.current_index = 0
        self.position = None  # None, 'BUY'
        self.entry_price = 0.0

    def next(self, bar):
        """
        Processes the next bar in the sequence.
        :param bar: A dictionary representing the current bar.
        :return: Signal dictionary.
        """
        idx = self.current_index
        self.current_index += 1
        
        # Need at least 'period' bars to calculate channel
        if idx < self.period:
            return {"type": "HOLD", "price": bar['close'], "confidence": 0.0, "metadata": {}}

        # Calculate channel based on PREVIOUS 'period' bars
        lookback_highs = self.high[idx-self.period:idx]
        lookback_lows = self.low[idx-self.period:idx]
        
        upper_channel = np.max(lookback_highs)
        lower_channel = np.min(lookback_lows)
        midline = (upper_channel + lower_channel) / 2
        channel_width = upper_channel - lower_channel
        
        current_close = bar['close']
        signal_type = "HOLD"
        confidence = 0.0
        metadata = {
            "upper_channel": upper_channel,
            "lower_channel": lower_channel,
            "midline": midline,
            "channel_width": channel_width
        }

        if self.position is None:
            if current_close > upper_channel:
                self.position = "BUY"
                self.entry_price = current_close
                signal_type = "BUY"
                # Confidence: breakout distance / channel width
                # Use a small epsilon for width to avoid division by zero
                dist = current_close - upper_channel
                confidence = dist / (channel_width if channel_width > 0 else 1e-9)
            elif current_close < lower_channel:
                # Shorting logic is not explicitly requested, but SELL usually implies exit or short.
                # Given the prompt "SELL when close < lowest low", we'll treat it as a signal.
                # However, the prompt also says "Exit when price touches channel midline".
                # I'll treat 'SELL' as a signal for exiting a long or entering a short.
                # For consistency with a single direction bot:
                self.position = "SELL"
                self.entry_price = current_close
                signal_type = "SELL"
                dist = lower_channel - current_close
                confidence = dist / (channel_width if channel_width > 0 else 1e-9)
        
        elif self.position == "BUY":
            if current_close < lower_channel:
                self.position = None
                signal_type = "SELL"
            elif current_close >= midline and current_close <= upper_channel:
                # Midline exit
                # Check if it actually "touches" or crosses.
                # If it's between midline and upper, and it was above, we exited.
                # But wait, if it was above upper, it's in a position. 
                # If it crosses the midline downwards, we exit.
                # The prompt says "Exit when price touches channel midline".
                # We'll check if the previous close was above midline and current is below, or vice versa.
                # For simplicity: if price is near midline.
                self.position = None
                signal_type = "SELL"
                confidence = 0.5 # Arbitrary exit confidence
        
        elif self.position == "SELL":
            if current_close > upper_channel:
                self.position = None
                signal_type = "BUY"
            elif current_close <= midline and current_close >= lower_channel:
                self.position = None
                signal_type = "BUY"
                confidence = 0.5

        return {"type": signal_type, "price": current_close, "confidence": float(confidence), "metadata": metadata}

    def get_signals(self):
        """Returns the history of signals (this implementation requires external tracking)."""
        # This is a stub as the main loop usually calls next()
        return []

    def get_params(self):
        """Returns the strategy parameters."""
        return {"period": self.period}
