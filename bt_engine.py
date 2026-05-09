"""
Backtest Engine — downloads real historical data, runs all strategies, stores results.
"""
import os
import json
import time
import numpy as np
from typing import Dict, List, Optional
import requests


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "backtest_results.json")
DATA_CACHE = os.path.join(os.path.dirname(__file__), "backtest_cache.npy")

SYMBOLS = {
    "BTC/USDT": "btcusdt",
    "ETH/USDT": "ethusdt",
    "SOL/USDT": "solusdt",
    "XRP/USDT": "xrpusdt",
}


def download_klines(symbol: str, interval: str = "1h", limit: int = 8760) -> np.ndarray:
    """Download klines from Binance. Returns numpy array [open, high, low, close, volume]."""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None

    while len(all_data) < limit:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(1000, limit - len(all_data)),
        }
        if end_time:
            params["endTime"] = end_time - 1

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_data = data + all_data  # prepend older data
        end_time = data[0][0]  # oldest timestamp

    # Parse to numpy: [open, high, low, close, volume]
    result = []
    for k in all_data[-limit:]:
        try:
            result.append([
                float(k[1]),   # open
                float(k[2]),   # high
                float(k[3]),   # low
                float(k[4]),   # close
                float(k[5]),   # volume
            ])
        except (ValueError, IndexError):
            continue

    return np.array(result, dtype=float) if result else np.empty((0, 5))


def run_backtest(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 8760) -> dict:
    """Run backtest on all strategies with real historical data."""
    from engine import StrategyEngine, create_strategies, _dict_data

    print(f"[Backtest] Downloading {symbol} {interval} data (limit={limit})...")
    candles = download_klines(symbol, interval, limit)

    if len(candles) < 100:
        return {"error": f"Insufficient data: {len(candles)} candles"}

    print(f"[Backtest] Got {len(candles)} candles. Running strategies...")

    # Cache data
    np.save(DATA_CACHE, candles)

    strategies = create_strategies()
    results = []

    for sw in strategies:
        try:
            sw.initialize(candles)
        except Exception as e:
            print(f"  Skip {sw.state.name} init: {e}")
            continue

        # Run through all bars
        equity = [10000.0]
        position = None
        entry_price = 0.0
        trades = 0
        wins = 0
        total_pnl = 0.0

        for i in range(60, len(candles)):  # skip warmup
            sig = None
            try:
                sig = sw.process_bar(candles, i)
            except Exception:
                continue

            if sig is None:
                continue

            sig_type = sig.get("type", "HOLD")
            confidence = sig.get("confidence", 0.0)
            price = float(candles[i, 3])

            if sig_type == "BUY" and confidence >= 0.1 and position is None:
                position = "LONG"
                entry_price = price
            elif sig_type == "SELL" and confidence >= 0.1 and position == "LONG":
                pnl_pct = (price - entry_price) / entry_price * 100
                total_pnl += pnl_pct
                trades += 1
                if pnl_pct > 0:
                    wins += 1
                equity.append(equity[-1] * (1 + pnl_pct / 100))
                position = None
                entry_price = 0.0

        # Calculate metrics
        eq = np.array(equity)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760))

        downside = returns[returns < 0]
        sortino = 0.0
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(8760))

        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        total_return = ((eq[-1] / eq[0]) - 1) * 100 if eq[0] > 0 else 0
        win_rate = (wins / trades * 100) if trades > 0 else 0
        profit_factor = 0.0
        # Simple approximation
        if trades > 0:
            avg_win = total_pnl / trades if trades > 0 else 0
            profit_factor = max(0, avg_win / (abs(total_pnl / trades) + 0.01))

        results.append({
            "strategy": sw.state.name,
            "category": sw.state.category,
            "total_return": round(total_return, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(win_rate, 1),
            "trades": trades,
            "profit_factor": round(profit_factor, 2),
            "equity_curve": equity[-200:],
        })

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    output = {
        "symbol": symbol,
        "interval": interval,
        "candles": len(candles),
        "timestamp": time.time(),
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Backtest] Complete. {len(results)} strategies tested.")
    return output


def get_backtest_results() -> Optional[dict]:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None
