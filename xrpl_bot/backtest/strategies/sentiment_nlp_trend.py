import numpy as np

class SentimentNLPTrend:
    \"\"\"
    Sentiment NLP Trend strategy.
    Synthetic sentiment proxy: 0.4*returns + 0.3*vol_change + 0.3*(RSI-50)/50.
    Smooth EMA window=100.
    BUY when sentiment > 0.5.
    \"\"\"

    def __init__(self, data):
        self.data = data
        self.window = 100
        self.current_index = 0
        
        # Extract series
        close = data['close'].values if hasattr(data, 'values') else np.array(data['close'])
        vol = data['volume'].values if hasattr(data, 'values') else np.array(data['volume'])
        
        # 1. Returns
        self.returns = np.diff(close, prepend=close[0])
        
        # 2. Vol Change (%)
        vol_change = np.diff(vol, prepend=vol[0]) / (vol + 1e-9)
        
        # 3. RSI (14)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = self._ema(gain, 14)
        avg_loss = self._ema(loss, 14)
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Raw Sentiment
        # 0.4*returns + 0.3*vol_change + 0.3*(RSI-50)/50
        # Note: we normalize returns/vol_change slightly for the proxy to work in [0,1] range
        # In a real scenario, these would be NLP-derived scores.
        raw_sentiment = (0.4 * self.returns + 
                         0.3 * vol_change + 
                         0.3 * (rsi - 50) / 50)
        
        # Smooth with EMA window 100
        self.sentiment_series = self._ema(raw_sentiment, self.window)
        
        self.prices = close
        self.signals = []

    def _ema(self, series, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

    def next(self, bar):
        idx = self.current_index
        price = bar['close']
        sentiment = self.sentiment_series[idx]
        
        signal = {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
        
        if sentiment > 0.5:
            signal = {
                "type": "BUY",
                "price": float(price),
                "confidence": float(min(sentiment, 1.0)),
                "metadata": {"sentiment": float(sentiment)}
            }
        elif sentiment < -0.5:
            signal = {
                "type": "SELL",
                "price": float(price),
                "confidence": float(min(abs(sentiment), 1.0)),
                "metadata": {"sentiment": float(sentiment)}
            }

        self.current_index += 1
        return signal

    def get_signals(self):
        return self.signals

    def get_params(self):
        return {"ema_window": self.window}
