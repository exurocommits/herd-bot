"""
Strategy Engine — wraps 20 strategies from xrpl_bot into a common interface.
"""
import sys
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Add xrpl_bot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xrpl_bot'))

from backtest.strategies.base_strategy import BaseStrategy, Signal

# ── Unified Strategy Wrapper ──────────────────────────────────────────────────

@dataclass
class StrategyState:
    name: str
    category: str
    description: str
    active: bool = True
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    equity_curve: List[float] = field(default_factory=lambda: [10000.0])
    last_signal: Optional[str] = None
    last_signal_time: float = 0.0
    position: Optional[str] = None  # 'LONG' or None
    entry_price: float = 0.0
    signals_history: List[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0

    @property
    def return_pct(self) -> float:
        if not self.equity_curve or self.equity_curve[0] == 0:
            return 0.0
        return ((self.equity_curve[-1] / self.equity_curve[0]) - 1) * 100

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 10:
            return 0.0
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for v in self.equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd


class StrategyWrapper:
    """Wraps any xrpl_bot strategy into our unified interface."""

    def __init__(self, name: str, category: str, description: str,
                 strategy_class, init_data_fn, extract_signal_fn):
        self.state = StrategyState(
            name=name, category=category, description=description
        )
        self._strategy_class = strategy_class
        self._init_data_fn = init_data_fn
        self._extract_signal_fn = extract_signal_fn
        self._instance = None
        self._warmup = 60  # bars needed before signals

    def initialize(self, candles: np.ndarray):
        """Create strategy instance with historical candle data."""
        try:
            data = self._init_data_fn(candles)
            self._instance = self._strategy_class(data)
        except Exception as e:
            print(f"[WARN] Init {self.state.name}: {e}")
            self._instance = None

    def process_bar(self, candles: np.ndarray, bar_idx: int):
        """Process a single bar and return signal dict."""
        if not self.state.active or self._instance is None:
            return None
        if bar_idx < self._warmup:
            return None
        try:
            sig = self._extract_signal_fn(self._instance, candles, bar_idx)
            return sig
        except Exception:
            return None

    def apply_signal(self, signal: Optional[dict], price: float, timestamp: float):
        """Apply signal to strategy state for paper trading."""
        if signal is None:
            return

        sig_type = signal.get("type", "HOLD")
        confidence = signal.get("confidence", 0.0)

        if sig_type == "HOLD" or confidence < 0.1:
            return

        # Record signal
        self.state.last_signal = sig_type
        self.state.last_signal_time = timestamp
        self.state.signals_history.append({
            "type": sig_type,
            "price": price,
            "confidence": confidence,
            "timestamp": timestamp,
        })
        if len(self.state.signals_history) > 200:
            self.state.signals_history = self.state.signals_history[-200:]

        # Paper trade execution
        if sig_type == "BUY" and self.state.position is None:
            self.state.position = "LONG"
            self.state.entry_price = price
        elif sig_type == "SELL" and self.state.position == "LONG":
            pnl = (price - self.state.entry_price) / self.state.entry_price * 100
            self.state.pnl += pnl
            self.state.trades += 1
            if pnl > 0:
                self.state.wins += 1
            # Update equity
            equity = self.state.equity_curve[-1] * (1 + pnl / 100)
            self.state.equity_curve.append(equity)
            if len(self.state.equity_curve) > 500:
                self.state.equity_curve = self.state.equity_curve[-500:]
            self.state.position = None
            self.state.entry_price = 0.0


# ── Data preparation helpers ──────────────────────────────────────────────────

def _dict_data(candles: np.ndarray) -> dict:
    """Convert candles array to dict format for strategies that need dict."""
    return {
        'close': candles[:, 3].astype(float),
        'high': candles[:, 1].astype(float),
        'low': candles[:, 2].astype(float),
        'open': candles[:, 0].astype(float),
        'volume': candles[:, 4].astype(float),
        'timestamp': np.arange(len(candles), dtype=float),
    }


def _close_array(candles: np.ndarray) -> np.ndarray:
    return candles[:, 3].astype(float)


def _base_strategy_data(candles: np.ndarray) -> dict:
    return {
        'close': candles[:, 3].astype(float),
        'high': candles[:, 1].astype(float),
        'low': candles[:, 2].astype(float),
        'open': candles[:, 0].astype(float),
        'volume': candles[:, 4].astype(float),
        'timestamp': np.arange(len(candles), dtype=float),
    }


# ── Signal extraction helpers ─────────────────────────────────────────────────

def _dict_next(instance, candles, bar_idx):
    """For strategies with .next(bar) interface."""
    try:
        return instance.next(bar_idx)
    except TypeError:
        price = float(candles[bar_idx, 3])
        try:
            return instance.next(bar_idx)
        except Exception:
            return {"type": "HOLD", "price": price, "confidence": 0.0}


def _base_next(instance, candles, bar_idx):
    """For strategies inheriting from BaseStrategy."""
    data = _base_strategy_data(candles)
    instance.data = data
    instance.current_index = bar_idx
    bar_dict = {
        'close': float(candles[bar_idx, 3]),
        'high': float(candles[bar_idx, 1]),
        'low': float(candles[bar_idx, 2]),
        'open': float(candles[bar_idx, 0]),
        'volume': float(candles[bar_idx, 4]),
        'timestamp': float(bar_idx),
    }
    sig = instance.next(bar_dict)
    if isinstance(sig, Signal):
        return {"type": sig.type, "price": sig.price, "confidence": sig.confidence}
    return {"type": "HOLD", "price": float(candles[bar_idx, 3]), "confidence": 0.0}


# ── Create all 20 strategy wrappers ───────────────────────────────────────────

def create_strategies() -> List[StrategyWrapper]:
    strategies = []

    try:
        from backtest.strategies.aroon_oscillator import AroonOscillator
        strategies.append(StrategyWrapper(
            "Aroon Oscillator", "Trend", "Detects trend direction using Aroon Up/Down",
            AroonOscillator, _close_array, _dict_next
        ))
    except Exception as e:
        print(f"Skip AroonOscillator: {e}")

    try:
        from backtest.strategies.cci_mean_reversion import CCIMeanReversion
        strategies.append(StrategyWrapper(
            "CCI Mean Reversion", "Mean Reversion", "Trades CCI extremes with mean reversion",
            CCIMeanReversion, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip CCIMeanReversion: {e}")

    try:
        from backtest.strategies.ema_cross_golden import EMACrossGolden
        strategies.append(StrategyWrapper(
            "EMA Golden Cross", "Trend", "Fast/slow EMA crossover with ATR confidence",
            EMACrossGolden, _base_strategy_data, _base_next
        ))
    except Exception as e:
        print(f"Skip EMACrossGolden: {e}")

    try:
        from backtest.strategies.rsi_divergence_play import RSIDivergencePlay
        strategies.append(StrategyWrapper(
            "RSI Divergence", "Mean Reversion", "Bullish/bearish RSI divergence detection",
            RSIDivergencePlay, _base_strategy_data, _base_next
        ))
    except Exception as e:
        print(f"Skip RSIDivergencePlay: {e}")

    try:
        from backtest.strategies.vortex_indicator import VortexIndicator
        strategies.append(StrategyWrapper(
            "Vortex Indicator", "Trend", "VI+/VI- crossover for trend detection",
            VortexIndicator, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip VortexIndicator: {e}")

    try:
        from backtest.strategies.adx_trend_strength import ADXTrendStrength
        strategies.append(StrategyWrapper(
            "ADX Trend Strength", "Trend", "ADX +DI/-DI directional movement",
            ADXTrendStrength, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip ADXTrendStrength: {e}")

    try:
        from backtest.strategies.keltner_channel_rev import KeltnerChannelRev
        strategies.append(StrategyWrapper(
            "Keltner Channel", "Mean Reversion", "EMA+ATR channel reversal trading",
            KeltnerChannelRev, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip KeltnerChannelRev: {e}")

    try:
        from backtest.strategies.linear_regression_channel import LinearRegressionChannel
        strategies.append(StrategyWrapper(
            "LinReg Channel", "Mean Reversion", "Linear regression channel with std dev bands",
            LinearRegressionChannel, _close_array, _dict_next
        ))
    except Exception as e:
        print(f"Skip LinearRegressionChannel: {e}")

    try:
        from backtest.strategies.ichimoku_cloud_breakout import IchimokuCloudBreakout
        strategies.append(StrategyWrapper(
            "Ichimoku Cloud", "Breakout", "Price/Tenkan/Kijun vs Ichimoku cloud",
            IchimokuCloudBreakout, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip IchimokuCloudBreakout: {e}")

    try:
        from backtest.strategies.bollinger_squeeze import BollingerSqueezeStrategy
        strategies.append(StrategyWrapper(
            "Bollinger Squeeze", "Breakout", "BB width squeeze detection with breakout signals",
            BollingerSqueezeStrategy, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip BollingerSqueezeStrategy: {e}")

    try:
        from backtest.strategies.choppiness_index import ChoppinessIndexStrategy
        strategies.append(StrategyWrapper(
            "Choppiness Index", "Trend", "Market regime detection (trending vs choppy)",
            ChoppinessIndexStrategy, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip ChoppinessIndexStrategy: {e}")

    try:
        from backtest.strategies.ehlers_supersmoother import EhlersSuperSmoother
        strategies.append(StrategyWrapper(
            "Ehlers Smoother", "Trend", "2-pole super smoother crossover signals",
            EhlersSuperSmoother, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip EhlersSuperSmoother: {e}")

    try:
        from backtest.strategies.weighted_close_reversion import WeightedCloseReversion
        strategies.append(StrategyWrapper(
            "Weighted Close Rev", "Mean Reversion", "Weighted close price mean reversion",
            WeightedCloseReversion, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip WeightedCloseReversion: {e}")

    try:
        from backtest.strategies.obv_divergence_signal import OBVDivergenceSignal
        strategies.append(StrategyWrapper(
            "OBV Divergence", "Volume", "On-Balance Volume divergence detection",
            OBVDivergenceSignal, _base_strategy_data, _base_next
        ))
    except Exception as e:
        print(f"Skip OBVDivergenceSignal: {e}")

    try:
        from backtest.strategies.macd_momentum_pure import MACDMomentumPure
        strategies.append(StrategyWrapper(
            "MACD Momentum", "Momentum", "MACD line/signal crossover with histogram confidence",
            MACDMomentumPure, _base_strategy_data, lambda inst, c, idx: _macd_next(inst, c, idx)
        ))
    except Exception as e:
        print(f"Skip MACDMomentumPure: {e}")

    try:
        from backtest.strategies.anomaly_detection import AnomalyDetectionSignal
        strategies.append(StrategyWrapper(
            "Anomaly Detection", "ML", "Mahalanobis distance anomaly detection",
            AnomalyDetectionSignal, _anomaly_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip AnomalyDetectionSignal: {e}")

    try:
        from backtest.strategies.distance_from_ma import DistanceFromMAStrategy
        strategies.append(StrategyWrapper(
            "Distance from MA", "Mean Reversion", "Z-score distance from 50 SMA",
            DistanceFromMAStrategy, _pandas_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip DistanceFromMAStrategy: {e}")

    try:
        from backtest.strategies.bollinger_bandwidth import BollingerBandwidthStrategy
        strategies.append(StrategyWrapper(
            "BB Width Expansion", "Breakout", "Bollinger Band width expansion/contraction",
            BollingerBandwidthStrategy, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip BollingerBandwidthStrategy: {e}")

    try:
        from backtest.strategies.n_bar_high_low import NBarHighLowStrategy
        strategies.append(StrategyWrapper(
            "N-Bar Breakout", "Breakout", "10-bar high/low breakout with trailing stop",
            NBarHighLowStrategy, _nbar_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip NBarHighLowStrategy: {e}")

    try:
        from backtest.strategies.ledger_momentum_scalp import LedgerMomentumScalp
        strategies.append(StrategyWrapper(
            "Ledger Momentum", "Scalping", "Transaction count momentum scalping",
            LedgerMomentumScalp, _dict_data, _dict_next
        ))
    except Exception as e:
        print(f"Skip LedgerMomentumScalp: {e}")

    return strategies


def _macd_next(instance, candles, bar_idx):
    data = _base_strategy_data(candles)
    instance.data = data
    try:
        from backtest.strategies.base_strategy import Signal
        bar_dict = {
            'close': float(candles[bar_idx, 3]),
            'timestamp': float(bar_idx),
        }
        sig = instance.next(bar_idx)
        if isinstance(sig, Signal):
            return {"type": sig.type, "price": sig.price, "confidence": sig.confidence}
    except Exception:
        pass
    return {"type": "HOLD", "price": float(candles[bar_idx, 3]), "confidence": 0.0}


def _anomaly_data(candles: np.ndarray):
    import pandas as pd
    d = _dict_data(candles)
    df = pd.DataFrame(d)
    return df


def _pandas_data(candles: np.ndarray):
    import pandas as pd
    d = _dict_data(candles)
    return pd.DataFrame(d)


def _nbar_data(candles: np.ndarray) -> np.ndarray:
    return candles.astype(float)


# ── Engine ────────────────────────────────────────────────────────────────────

class StrategyEngine:
    """Orchestrates all strategy wrappers."""

    def __init__(self):
        self.strategies: List[StrategyWrapper] = create_strategies()
        self.signal_feed: List[dict] = []  # recent signals for the dashboard

    def initialize_all(self, candles: np.ndarray):
        for sw in self.strategies:
            sw.initialize(candles)
        print(f"[Engine] Initialized {len(self.strategies)} strategies")

    def process_tick(self, candles: np.ndarray, bar_idx: int, prices: Dict[str, float]):
        """Process all strategies on a new bar."""
        price = prices.get("BTC/USDT", 0)
        timestamp = time.time()
        new_signals = []

        for sw in self.strategies:
            sig = sw.process_bar(candles, bar_idx)
            if sig and sig.get("type") != "HOLD":
                sig["strategy"] = sw.state.name
                sig["asset"] = "BTC/USDT"
                sig["timestamp"] = timestamp
                new_signals.append(sig)
                self.signal_feed.append(sig)

            sw.apply_signal(sig, price, timestamp)

        # Keep signal feed manageable
        if len(self.signal_feed) > 500:
            self.signal_feed = self.signal_feed[-500:]

        return new_signals

    def get_strategies_summary(self) -> List[dict]:
        results = []
        for sw in self.strategies:
            s = sw.state
            results.append({
                "name": s.name,
                "category": s.category,
                "description": s.description,
                "active": s.active,
                "pnl": round(s.pnl, 2),
                "return_pct": round(s.return_pct, 2),
                "trades": s.trades,
                "win_rate": round(s.win_rate, 1),
                "sharpe": round(s.sharpe, 2),
                "max_drawdown": round(s.max_drawdown, 2),
                "last_signal": s.last_signal,
                "position": s.position,
                "equity_curve": s.equity_curve[-100:],
            })
        return results

    def start_strategy(self, name: str) -> bool:
        for sw in self.strategies:
            if sw.state.name == name:
                sw.state.active = True
                return True
        return False

    def stop_strategy(self, name: str) -> bool:
        for sw in self.strategies:
            if sw.state.name == name:
                sw.state.active = False
                return True
        return False

    def get_performance(self) -> dict:
        active = [sw for sw in self.strategies if sw.state.active]
        total_pnl = sum(sw.state.pnl for sw in active)
        total_trades = sum(sw.state.trades for sw in active)
        total_wins = sum(sw.state.wins for sw in active)
        equity_curves = [sw.state.equity_curve for sw in active if len(sw.state.equity_curve) > 1]
        combined_equity = []
        if equity_curves:
            min_len = min(len(e) for e in equity_curves)
            for i in range(min_len):
                combined_equity.append(sum(e[-(min_len - i)] for e in equity_curves) / len(equity_curves))

        return {
            "total_strategies": len(self.strategies),
            "active_strategies": len(active),
            "total_pnl": round(total_pnl, 2),
            "total_trades": total_trades,
            "overall_win_rate": round((total_wins / total_trades * 100) if total_trades > 0 else 0, 1),
            "combined_equity_curve": combined_equity[-100:],
        }
