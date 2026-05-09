"""
Herd Sentiment Analyzer
Simulates social sentiment, tracks herd momentum, generates contrarian signals.
Enhanced with HyperLiquid market signals: funding rate, OI, order book.
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
    # HL-specific
    funding_signal: Optional[str] = None   # contrarian funding signal
    funding_confidence: float = 0.0
    oi_signal: Optional[str] = None       # OI momentum signal
    oi_confidence: float = 0.0
    book_imbalance: float = 0.0           # -1 to +1 (bid-heavy vs ask-heavy)
    liquidation_risk: float = 0.0         # 0 to 1


class HerdSentimentAnalyzer:
    """Tracks social sentiment, herd momentum, whale movements, fear/greed, and HL market signals."""

    ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "HYPE/USDT"]

    def __init__(self):
        self.sentiment: Dict[str, SentimentData] = {}
        self.history: Dict[str, List[Tuple[float, float]]] = {}
        self.max_history = 500
        self._oi_history: Dict[str, List[float]] = {}  # coin -> recent OI values
        self._init_sentiment()

    def _init_sentiment(self):
        for asset in self.ASSETS:
            base_scores = {
                "BTC/USDT": 25.0,
                "ETH/USDT": 20.0,
                "SOL/USDT": 35.0,
                "HYPE/USDT": 15.0,
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
            coin = asset.split("/")[0]
            self._oi_history[coin] = []

    def update(self, prices: Dict[str, float], price_changes: Dict[str, float]):
        """Update sentiment based on price movements (simulated social reaction)."""
        now = time.time()

        for asset in self.ASSETS:
            sd = self.sentiment[asset]
            price_change = price_changes.get(asset, 0.0)

            # Simulate social media reaction to price changes
            social_reaction = price_change * 15 + random.gauss(0, 3)

            # Mean-reverting random walk
            reversion = -sd.score * 0.08

            # Random events
            if random.random() < 0.05:
                social_reaction += random.gauss(0, 25)

            new_score = sd.score + social_reaction + reversion
            new_score = max(-100.0, min(100.0, new_score))

            momentum = new_score - sd.score

            # Social volume
            base_volume = {"BTC/USDT": 3000, "ETH/USDT": 2000, "SOL/USDT": 1500, "HYPE/USDT": 1200}
            bv = base_volume.get(asset, 1000)
            volume = int(bv + abs(momentum) * 100 + random.randint(-100, 100))
            volume = max(100, volume)

            # Whale activity (simulated)
            whale_alerts = sd.whale_activity[-5:]
            if random.random() < 0.08:
                direction = "BUY" if new_score > sd.score else "SELL"
                sizes = ["🐋 Large", "🐋 Whale", "🐋 Mega"]
                amounts = ["500 BTC", "1000 ETH", "5000 SOL", "500K HYPE"]
                amount = random.choice(amounts)
                whale_alerts.append(f"{random.choice(sizes)} {direction}: {amount}")

            # Fear & Greed index
            fear_greed = 50.0 + new_score * 0.35 + random.gauss(0, 3)
            fear_greed = max(0.0, min(100.0, fear_greed))

            # Contrarian signals (social sentiment extremes)
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

            self.history[asset].append((now, new_score))
            if len(self.history[asset]) > self.max_history:
                self.history[asset] = self.history[asset][-self.max_history:]

    def update_hl_signals(
        self,
        funding_rates: Dict[str, dict],
        open_interest: Dict[str, dict],
        order_books: Dict[str, dict] = None,
    ):
        """
        Update sentiment with HyperLiquid market signals.
        - funding_rates: {coin: {"rate": float, "timestamp": int}}
        - open_interest: {coin: dict from HL API}
        - order_books: {coin: {"bids": [...], "asks": [...]}}
        """
        for asset in self.ASSETS:
            coin = asset.split("/")[0]
            sd = self.sentiment[asset]

            # ── Funding Rate Signal ──────────────────────────────────────
            funding_signal = None
            funding_confidence = 0.0
            fr_data = funding_rates.get(coin, {})
            rate = fr_data.get("rate", 0.0) if isinstance(fr_data, dict) else 0.0

            if abs(rate) > 0.0001:  # non-trivial funding
                # Extreme positive funding = crowd is long → contrarian SELL
                # Extreme negative funding = crowd is short → contrarian BUY
                annualized = rate * 3 * 365  # approx annual (3 funding periods/day)
                if annualized > 50:  # >50% annual = extreme long
                    funding_signal = "SELL"
                    funding_confidence = min(1.0, annualized / 200)
                elif annualized < -50:  # <-50% annual = extreme short
                    funding_signal = "BUY"
                    funding_confidence = min(1.0, abs(annualized) / 200)

            # ── Open Interest Signal ─────────────────────────────────────
            oi_signal = None
            oi_confidence = 0.0
            oi_data = open_interest.get(coin, {})
            oi_value = 0.0
            if isinstance(oi_data, dict):
                oi_value = float(oi_data.get("openInterest", oi_data.get("sz", 0)))

            self._oi_history.setdefault(coin, []).append(oi_value)
            if len(self._oi_history[coin]) > 50:
                self._oi_history[coin] = self._oi_history[coin][-50:]

            # OI momentum: rapid increase = new positions entering
            oi_hist = self._oi_history[coin]
            if len(oi_hist) >= 5 and oi_hist[0] > 0:
                oi_change = (oi_hist[-1] - oi_hist[-5]) / oi_hist[-5]
                price = self.sentiment[asset].score  # use sentiment direction as price proxy

                if oi_change > 0.1 and price > 0:
                    # OI up + price up = strong trend → BUY
                    oi_signal = "BUY"
                    oi_confidence = min(1.0, oi_change)
                elif oi_change > 0.1 and price < 0:
                    # OI up + price down = potential squeeze → watch
                    oi_signal = "SELL"
                    oi_confidence = min(1.0, oi_change * 0.5)
                elif oi_change < -0.1:
                    # OI dropping = liquidation cascade
                    oi_signal = "SELL"
                    oi_confidence = min(1.0, abs(oi_change) * 0.3)

            # ── Order Book Imbalance ─────────────────────────────────────
            book_imbalance = 0.0
            if order_books and coin in order_books:
                book = order_books[coin]
                levels = book.get("levels", [])
                if len(levels) >= 2:
                    bids = levels[0] if isinstance(levels[0], list) else []
                    asks = levels[1] if isinstance(levels[1], list) else []
                    bid_vol = sum(float(b.get("sz", 0)) for b in bids[:10])
                    ask_vol = sum(float(a.get("sz", 0)) for a in asks[:10])
                    total = bid_vol + ask_vol
                    if total > 0:
                        book_imbalance = (bid_vol - ask_vol) / total

            # ── Liquidation Risk ─────────────────────────────────────────
            liquidation_risk = 0.0
            if len(oi_hist) >= 3 and oi_hist[-3] > 0:
                oi_drop = (oi_hist[-1] - oi_hist[-3]) / oi_hist[-3]
                if oi_drop < -0.05:  # OI dropped >5% quickly
                    liquidation_risk = min(1.0, abs(oi_drop) * 5)

            # Update the sentiment data with HL signals
            sd.funding_signal = funding_signal
            sd.funding_confidence = funding_confidence
            sd.oi_signal = oi_signal
            sd.oi_confidence = oi_confidence
            sd.book_imbalance = book_imbalance
            sd.liquidation_risk = liquidation_risk

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
                    "funding_signal": sd.funding_signal,
                    "funding_confidence": sd.funding_confidence,
                    "oi_signal": sd.oi_signal,
                    "oi_confidence": sd.oi_confidence,
                    "book_imbalance": sd.book_imbalance,
                    "liquidation_risk": sd.liquidation_risk,
                }
                for asset, sd in self.sentiment.items()
            },
        }
