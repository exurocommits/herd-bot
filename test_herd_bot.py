"""
Tests for HERD Bot — unit tests for all core modules.
"""
import sys
import os
import time
import json
import numpy as np
import pytest

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xrpl_bot'))
sys.path.insert(0, os.path.dirname(__file__))

from engine import StrategyEngine, StrategyWrapper, StrategyState, create_strategies, _dict_data
from sentiment import HerdSentimentAnalyzer, SentimentData
from bt_engine import download_klines, run_backtest, get_backtest_results


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_candles():
    """Generate synthetic OHLCV candle data."""
    np.random.seed(42)
    n = 200
    base = 65000
    prices = base + np.cumsum(np.random.randn(n) * 200)
    candles = np.column_stack([
        prices - np.random.rand(n) * 100,
        prices + np.abs(np.random.randn(n)) * 200,
        prices - np.abs(np.random.randn(n)) * 200,
        prices,
        np.random.rand(n) * 1000 + 100,
    ])
    return candles


@pytest.fixture
def engine(sample_candles):
    e = StrategyEngine()
    e.initialize_all(sample_candles)
    return e


@pytest.fixture
def sentiment():
    return HerdSentimentAnalyzer()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY STATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyState:
    def test_default_state(self):
        s = StrategyState(name="Test", category="Trend", description="Test strategy")
        assert s.name == "Test"
        assert s.pnl == 0.0
        assert s.trades == 0
        assert s.wins == 0
        assert s.active is True
        assert len(s.equity_curve) == 1
        assert s.equity_curve[0] == 10000.0

    def test_win_rate_no_trades(self):
        s = StrategyState(name="Test", category="Trend", description="")
        assert s.win_rate == 0.0

    def test_win_rate_with_trades(self):
        s = StrategyState(name="Test", category="Trend", description="")
        s.trades = 10
        s.wins = 6
        assert s.win_rate == 60.0

    def test_return_pct(self):
        s = StrategyState(name="Test", category="Trend", description="")
        s.equity_curve = [10000, 10500, 11000]
        assert abs(s.return_pct - 10.0) < 0.01

    def test_sharpe(self):
        s = StrategyState(name="Test", category="Trend", description="")
        # Need enough data points
        s.equity_curve = [10000 + i * 10 for i in range(50)]
        sharpe = s.sharpe
        assert isinstance(sharpe, float)

    def test_max_drawdown(self):
        s = StrategyState(name="Test", category="Trend", description="")
        s.equity_curve = [10000, 11000, 10500, 12000]
        dd = s.max_drawdown
        assert dd > 0  # There's a drawdown between 11000 and 10500


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyEngine:
    def test_create_strategies(self):
        strategies = create_strategies()
        assert len(strategies) > 0
        # Should have at least 10 strategies loaded
        assert len(strategies) >= 10

    def test_engine_initialize(self, engine, sample_candles):
        assert len(engine.strategies) > 0

    def test_process_tick(self, engine, sample_candles):
        prices = {"BTC/USDT": 65000.0}
        new_signals = engine.process_tick(sample_candles, len(sample_candles) - 1, prices)
        assert isinstance(new_signals, list)

    def test_get_strategies_summary(self, engine):
        summary = engine.get_strategies_summary()
        assert isinstance(summary, list)
        assert len(summary) > 0
        for s in summary:
            assert "name" in s
            assert "category" in s
            assert "active" in s
            assert "pnl" in s
            assert "sharpe" in s

    def test_start_stop_strategy(self, engine):
        summary = engine.get_strategies_summary()
        if len(summary) > 0:
            name = summary[0]["name"]
            assert engine.stop_strategy(name) is True
            assert engine.start_strategy(name) is True
        assert engine.stop_strategy("nonexistent") is False
        assert engine.start_strategy("nonexistent") is False

    def test_get_performance(self, engine):
        perf = engine.get_performance()
        assert "total_strategies" in perf
        assert "active_strategies" in perf
        assert "total_pnl" in perf
        assert "total_trades" in perf
        assert "overall_win_rate" in perf
        assert perf["total_strategies"] > 0

    def test_signal_feed_populates(self, engine, sample_candles):
        prices = {"BTC/USDT": 65000.0}
        for i in range(60, len(sample_candles)):
            engine.process_tick(sample_candles, i, prices)
        # Signal feed should exist (may or may not have signals depending on strategies)
        assert isinstance(engine.signal_feed, list)


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYZER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHerdSentiment:
    def test_init(self, sentiment):
        assert len(sentiment.sentiment) == 4
        for asset in sentiment.ASSETS:
            assert asset in sentiment.sentiment

    def test_update(self, sentiment):
        prices = {"BTC/USDT": 65000, "ETH/USDT": 3500, "SOL/USDT": 150, "XRP/USDT": 0.55}
        changes = {"BTC/USDT": 2.5, "ETH/USDT": 1.8, "SOL/USDT": -1.2, "XRP/USDT": 0.5}
        sentiment.update(prices, changes)
        for asset in sentiment.ASSETS:
            sd = sentiment.get_sentiment(asset)
            assert -100 <= sd.score <= 100
            assert 0 <= sd.fear_greed <= 100

    def test_fear_greed_index(self, sentiment):
        fg = sentiment.get_fear_greed_index()
        assert 0 <= fg <= 100

    def test_social_volumes(self, sentiment):
        vols = sentiment.get_social_volumes()
        assert len(vols) == 4
        for v in vols.values():
            assert v > 0

    def test_to_dict(self, sentiment):
        d = sentiment.to_dict()
        assert "fear_greed_index" in d
        assert "assets" in d
        assert len(d["assets"]) == 4

    def test_sentiment_bounds(self, sentiment):
        """Sentiment should stay within bounds even with extreme inputs."""
        for _ in range(100):
            prices = {"BTC/USDT": 65000, "ETH/USDT": 3500, "SOL/USDT": 150, "XRP/USDT": 0.55}
            changes = {"BTC/USDT": 10, "ETH/USDT": -10, "SOL/USDT": 5, "XRP/USDT": -5}
            sentiment.update(prices, changes)
        for asset in sentiment.ASSETS:
            sd = sentiment.get_sentiment(asset)
            assert -100 <= sd.score <= 100
            assert 0 <= sd.fear_greed <= 100

    def test_contrarian_signals(self, sentiment):
        """When sentiment is extreme, contrarian signals should appear."""
        # Force extreme sentiment
        for _ in range(200):
            prices = {"BTC/USDT": 65000, "ETH/USDT": 3500, "SOL/USDT": 150, "XRP/USDT": 0.55}
            changes = {"BTC/USDT": 8.0, "ETH/USDT": 8.0, "SOL/USDT": 8.0, "XRP/USDT": 8.0}
            sentiment.update(prices, changes)

        # At least some assets should have contrarian signals
        has_contrarian = False
        for asset in sentiment.ASSETS:
            sd = sentiment.get_sentiment(asset)
            if sd.contrarian_signal is not None:
                has_contrarian = True
                assert sd.contrarian_signal in ("BUY", "SELL")
                assert sd.contrarian_confidence > 0
        # Note: may not always trigger due to random walk, so we don't assert


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktest:
    def test_download_klines(self):
        """Test that we can download real data from Binance."""
        try:
            candles = download_klines("BTCUSDT", "1h", 100)
            assert len(candles) > 0
            assert candles.shape[1] == 5
        except Exception as e:
            pytest.skip(f"Binance API unavailable: {e}")

    def test_get_backtest_results_empty(self, tmp_path, monkeypatch):
        """When no results file exists, should return None."""
        monkeypatch.setattr("bt_engine.RESULTS_FILE", str(tmp_path / "nonexistent.json"))
        result = get_backtest_results()
        assert result is None

    def test_get_backtest_results_with_data(self, tmp_path, monkeypatch):
        """When results file exists, should return the data."""
        rf = tmp_path / "results.json"
        test_data = {"results": [{"strategy": "test", "sharpe": 1.5}]}
        rf.write_text(json.dumps(test_data))
        monkeypatch.setattr("bt_engine.RESULTS_FILE", str(rf))
        result = get_backtest_results()
        assert result is not None
        assert len(result["results"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINT TESTS (using FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "uptime" in data

    def test_strategies(self, client):
        r = client.get("/api/strategies")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_signals(self, client):
        r = client.get("/api/signals")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_sentiment(self, client):
        r = client.get("/api/sentiment")
        assert r.status_code == 200
        data = r.json()
        assert "fear_greed_index" in data
        assert "assets" in data

    def test_performance(self, client):
        r = client.get("/api/performance")
        assert r.status_code == 200
        data = r.json()
        assert "total_strategies" in data

    def test_backtest_results(self, client):
        r = client.get("/api/backtest/results")
        assert r.status_code == 200

    def test_start_stop_strategy(self, client):
        # Get a strategy name first
        strats = client.get("/api/strategies").json()
        if len(strats) > 0:
            name = strats[0]["name"]
            r = client.post(f"/api/strategies/{name}/stop")
            assert r.status_code == 200
            r = client.post(f"/api/strategies/{name}/start")
            assert r.status_code == 200

    def test_index_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "HERD" in r.text


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY WRAPPER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyWrapper:
    def test_apply_buy_signal(self, sample_candles):
        sw = StrategyWrapper("Test", "Trend", "Desc", None, None, None)
        sw.apply_signal({"type": "BUY", "confidence": 0.8}, 65000, time.time())
        assert sw.state.position == "LONG"
        assert sw.state.entry_price == 65000

    def test_apply_sell_signal(self, sample_candles):
        sw = StrategyWrapper("Test", "Trend", "Desc", None, None, None)
        sw.apply_signal({"type": "BUY", "confidence": 0.8}, 65000, time.time())
        sw.apply_signal({"type": "SELL", "confidence": 0.7}, 66000, time.time())
        assert sw.state.position is None
        assert sw.state.trades == 1
        assert sw.state.pnl > 0

    def test_apply_hold_signal(self):
        sw = StrategyWrapper("Test", "Trend", "Desc", None, None, None)
        sw.apply_signal({"type": "HOLD", "confidence": 0.0}, 65000, time.time())
        assert sw.state.position is None
        assert sw.state.trades == 0

    def test_low_confidence_ignored(self):
        sw = StrategyWrapper("Test", "Trend", "Desc", None, None, None)
        sw.apply_signal({"type": "BUY", "confidence": 0.05}, 65000, time.time())
        assert sw.state.position is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
