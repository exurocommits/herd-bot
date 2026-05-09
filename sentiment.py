"""
Herd Sentiment Analyzer
Simulates social sentiment, tracks herd momentum, generates contrarian signals.
"""
import time
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SentimentData:
    asset: str
    score: float          # -100 to +100
    momentum: float       # rate of change
    social_volume: int    # mentions count
    whale_activity: List[str]  # recent whale alerts
    fear_greed: float     # 0-100 (0=extreme fear, 100=extreme greed)
    contrarian_signal: Optional[str] = None  # 'BUY' or 'SELL' when herd is extreme
    contrarian_confidence: float = 0.0
    timestamp: float = 0.0


class HerdSentimentAnalyzer:
    """Tracks social sentiment, herd momentum, whale movements, and fear/greed."""

    ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

    def __init__(self):
        self.sentiment: Dict[str, SentimentData] = {}
        self.history: Dict[str, List[Tuple[float, float]]] = {}  # asset -> [(ts, score)]
        self.max_history = 500
        self._init_sentiment()

    def _init_sentiment(self):
        for asset in self.ASSETS:
            base_scores = {
                "BTC/USDT": 25.0,
                "ETH/USDT": 20.0,
                "SOL/USDT": 35.0,
                "XRP/USDT": 15.0,
            }
            score = base_scores.get(asset, 0.0)
            self.sentiment[asset] = SentimentData(
                asset=asset,
                score=score,
                momentum=0.0,
                social_volume=random.randint(500, 2000),
                whale_activity=[],
                fear_greed=50.0 + score * 0.3,
                timestamp=time.time(),
            )
            self.history[asset] = [(time.time(), score)]

    def update(self, prices: Dict[str, float], price_changes: Dict[str, float]):
        """Update sentiment based on price movements (simulated social reaction)."""
        now = time.time()

        for asset in self.ASSETS:
            sd = self.sentiment[asset]
            price_change = price_changes.get(asset, 0.0)

            # Simulate social media reaction to price changes
            # Price up -> positive sentiment shift, price down -> negative
            social_reaction = price_change * 15 + random.gauss(0, 3)

            # Mean-reverting random walk (sentiment tends back to center)
            reversion = -sd.score * 0.08

            # Random events (news, hype, FUD)
            if random.random() < 0.05:  # 5% chance of "event"
                social_reaction += random.gauss(0, 25)

            new_score = sd.score + social_reaction + reversion
            new_score = max(-100.0, min(100.0, new_score))

            # Calculate momentum (rate of change)
            momentum = new_score - sd.score

            # Social volume correlates with |momentum|
            base_volume = {"BTC/USDT": 3000, "ETH/USDT": 2000, "SOL/USDT": 1500, "XRP/USDT": 800}
            bv = base_volume.get(asset, 1000)
            volume = int(bv + abs(momentum) * 100 + random.randint(-100, 100))
            volume = max(100, volume)

            # Generate whale activity (simulated)
            whale_alerts = sd.whale_activity[-5:]
            if random.random() < 0.08:
                direction = "BUY" if new_score > sd.score else "SELL"
                sizes = ["🐋 Large", "🐋 Whale", "🐋 Mega"]
                amount = random.choice(["500 BTC", "1000 ETH", "5000 SOL", "1M XRP"])
                whale_alerts.append(f"{random.choice(sizes)} {direction}: {amount}")

            # Fear & Greed index (composite)
            fear_greed = 50.0 + new_score * 0.35 + random.gauss(0, 3)
            fear_greed = max(0.0, min(100.0, fear_greed))

            # Contrarian signals
            contrarian_signal = None
            contrarian_confidence = 0.0
            if new_score > 80:
                contrarian_signal = "SELL"
                contrarian_confidence = min(1.0, (new_score - 80) / 20.0)
            elif new_score < -80:
                contrarian_signal = "BUY"
                contrarian_confidence = min(1.0, (-80 - new_score) / 20.0)

            self.sentiment[asset] = SentimentData(
                asset=asset,
                score=new_score,
                momentum=momentum,
                social_volume=volume,
                whale_activity=whale_alerts[-5:],
                fear_greed=fear_greed,
                contrarian_signal=contrarian_signal,
                contrarian_confidence=contrarian_confidence,
                timestamp=now,
            )

            # Store history
            self.history[asset].append((now, new_score))
            if len(self.history[asset]) > self.max_history:
                self.history[asset] = self.history[asset][-self.max_history:]

    def get_sentiment(self, asset: str) -> SentimentData:
        return self.sentiment.get(asset, SentimentData(
            asset=asset, score=0, momentum=0, social_volume=0,
            whale_activity=[], fear_greed=50, timestamp=time.time()
        ))

    def get_all_sentiment(self) -> Dict[str, SentimentData]:
        return dict(self.sentiment)

    def get_fear_greed_index(self) -> float:
        """Aggregate Fear & Greed across all assets."""
        if not self.sentiment:
            return 50.0
        return sum(s.fear_greed for s in self.sentiment.values()) / len(self.sentiment)

    def get_social_volumes(self) -> Dict[str, int]:
        return {asset: sd.social_volume for asset, sd in self.sentiment.items()}

    def to_dict(self) -> dict:
        return {
            "fear_greed_index": self.get_fear_greed_index(),
            "assets": {
                asset: {
                    "score": sd.score,
                    "momentum": sd.momentum,
                    "social_volume": sd.social_volume,
                    "whale_activity": sd.whale_activity,
                    "fear_greed": sd.fear_greed,
                    "contrarian_signal": sd.contrarian_signal,
                    "contrarian_confidence": sd.contrarian_confidence,
                    "timestamp": sd.timestamp,
                }
                for asset, sd in self.sentiment.items()
            },
        }
