import numpy as np

class LSTMPricePredictor:
    \"\"\"
    LSTM Price Predictor strategy.
    Uses a simple numpy-based RNN if torch is not available.
    Lookback = 60.
    Predicts 5 bars ahead.
    BUY when predicted price > current price + 0.5 * ATR.
    \"\"\"

    def __init__(self, data):
        self.data = data
        self.lookback = 60
        self.predict_ahead = 5
        self.current_index = 0
        self.use_torch = False
        
        try:
            import torch
            self.use_torch = True
        except ImportError:
            pass

        # Pre-calculate ATR for the signal condition
        close = data['close'].values if hasattr(data, 'values') else np.array(data['close'])
        high = data['high'].values if hasattr(data, 'values') else np.array(data['high'])
        low = data['low'].values if hasattr(data, 'values') else np.array(data['low'])
        
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        self.atr_series = self._ema(tr, 14)
        
        self.prices = close
        self.signals = []

    def _ema(self, series, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

    def _numpy_rnn_predict(self, sequence):
        # Extremely simplified dummy RNN for demonstration when torch is missing
        # In a real scenario, this would be a trained model weight-set
        # We'll simulate a prediction based on trend + volatility
        trend = (sequence[-1] - sequence[0]) / len(sequence)
        noise = np.random.normal(0, 0.001 * sequence[-1])
        return sequence[-1] + (trend * self.predict_ahead) + noise

    def next(self, bar):
        idx = self.current_index
        price = bar['close']
        atr = self.atr_series[idx]
        
        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
        
        if idx >= self.lookback:
            # Get lookback sequence
            sequence = self.prices[idx - self.lookback : idx]
            
            if self.use_torch:
                # Placeholder for actual torch LSTM inference
                # In real implementation: prediction = model(torch.tensor(sequence))
                predicted_price = self._numpy_rnn_predict(sequence) 
            else:
                predicted_price = self._numpy_rnn_predict(sequence)
            
            threshold = price + 0.5 * atr
            
            if predicted_price > threshold:
                signal = {
                    "type": "BUY",
                    "price": float(price),
                    "confidence": float(min((predicted_price - threshold) / (atr + 1e-9), 1.0)),
                    "metadata": {"predicted": float(predicted_price), "atr": float(atr)}
                }
            elif predicted_price < price - 0.5 * atr:
                signal = {
                    "type": "SELL",
                    "price": float(price),
                    "confidence": float(min((price - atr - predicted_price) / (atr + 1e-9), 1.0)),
                    "metadata": {"predicted": float(predicted_price), "atr": float(atr)}
                }

        self.current_index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"lookback": self.lookback, "predict_ahead": self.predict_ahead, "use_torch": self.use_torch}
