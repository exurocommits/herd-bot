import numpy as np

class TechnicalSentimentFusion:
    """
    Fuses technical indicators and synthetic sentiment scores.
    Technical score = (RSI-50)/50 + (MACD_hist)/ATR + BB%B.
    Sentiment score = 0.4*return + 0.3*vol_change + 0.3*(RSI-50)/50.
    Fused = 0.6*technical + 0.4*sentiment.
    """
    def __init__(self, data):
        self.data = data
        self.current_index = 0

    def _calculate_technical(self, idx):
        rsi = self.data['rsi'].iloc[idx]
        macd_hist = self.data['macd_hist'].iloc[idx] if 'macd_hist' in self.data.columns else 0.0
        atr = self.data['atr'].iloc[idx] if 'atr' in self.data.columns else 1.0
        # BB%B: (Price - LowerBand) / (UpperBand - LowerBand)
        bb_p_b = self.data['bb_p_b'].iloc[idx] if 'bb_p_b' in self.data.columns else 0.5
        
        return (rsi - 50) / 50 + (macd_hist / (atr + 1e-9)) + bb_p_b

    def _calculate_sentiment(self, idx):
        ret = self.data['close'].pct_change().iloc[idx] if idx > 0 else 0.0
        vol_change = self.data['volume'].pct_change().iloc[idx] if idx > 0 else 0.0
        rsi = self.data['rsi'].iloc[idx]
        
        return 0.4 * ret + 0.3 * vol_change + 0.3 * (rsi - 50) / 50

    def next(self, bar):
        idx = self.current_index
        price = self.data['close'].iloc[idx]
        
        tech = self._calculate_technical(idx)
        sent = self._calculate_sentiment(idx)
        fused = 0.6 * tech + 0.4 * sent
        
        self.current_index += 1
        return {"price": price, "fused": fused}

    def get_signals(self):
        return []

    def get_params(self):
        return {"tech_weight": 0.6, "sent_weight": 0.4}

    def get_signal_dict(self, fused, price):
        if fused > 0.4:
            return {"type": "BUY", "price": float(price), "confidence": float(abs(fused)), "metadata": {}}
        elif fused < -0.4:
            return {"type": "SELL", "price": float(price), "confidence": float(abs(fused)), "metadata": {}}
        return {"type": "HOLD", "price": float(price), "confidence": 0.0, "metadata": {}}
