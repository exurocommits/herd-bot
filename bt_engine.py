"""
Backtest Engine — downloads HL historical data, runs all strategies, portfolio optimization.
"""
import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import product as iterproduct

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "backtest_results.json")
DATA_CACHE = os.path.join(os.path.dirname(__file__), "backtest_cache.npy")

COINS = ["BTC", "ETH", "SOL", "HYPE"]


def download_hl_candles(
    coin: str = "BTC",
    interval: str = "1h",
    limit: int = 8760,
) -> np.ndarray:
    """Download candles from HyperLiquid. Returns numpy array [open, high, low, close, volume]."""
    from hl_client import hl_client

    end_time = int(time.time() * 1000)
    # Estimate start time based on interval
    interval_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
    }
    ms = interval_ms.get(interval, 3_600_000)
    start_time = end_time - limit * ms

    candles = hl_client.fetch_candle_snapshot(
        coin=coin,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
    )

    if len(candles) > limit:
        candles = candles[-limit:]

    return candles


# Keep old name for backward compat
def download_klines(symbol: str, interval: str = "1h", limit: int = 8760) -> np.ndarray:
    """Alias: download candles from HL. Symbol param kept for compat (coin extracted)."""
    coin = symbol.upper().replace("USDT", "").replace("/USDT", "")
    return download_hl_candles(coin=coin, interval=interval, limit=limit)


def run_backtest(
    symbol: str = "BTC",
    interval: str = "1h",
    limit: int = 8760,
) -> dict:
    """Run backtest on all strategies with real historical data."""
    from engine import StrategyEngine, create_strategies

    coin = symbol.upper().replace("USDT", "").replace("/USDT", "")
    print(f"[Backtest] Downloading {coin} {interval} data (limit={limit})...")
    candles = download_hl_candles(coin, interval, limit)

    if len(candles) < 100:
        return {"error": f"Insufficient data: {len(candles)} candles"}

    print(f"[Backtest] Got {len(candles)} candles. Running strategies...")

    np.save(DATA_CACHE, candles)

    strategies = create_strategies()
    results = []

    for sw in strategies:
        try:
            sw.initialize(candles)
        except Exception as e:
            print(f"  Skip {sw.state.name} init: {e}")
            continue

        equity = [10000.0]
        position = None
        entry_price = 0.0
        trades = 0
        wins = 0
        total_pnl = 0.0

        for i in range(60, len(candles)):
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

        metrics = _calc_metrics(equity, trades, wins, total_pnl)
        results.append({
            "strategy": sw.state.name,
            "category": sw.state.category,
            **metrics,
            "equity_curve": equity[-200:],
        })

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    output = {
        "symbol": coin,
        "interval": interval,
        "candles": len(candles),
        "timestamp": time.time(),
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Backtest] Complete. {len(results)} strategies tested.")
    return output


def run_portfolio_backtest(
    coins: List[str] = None,
    interval: str = "1h",
    limit: int = 4380,
    top_n: int = 5,
) -> dict:
    """
    Portfolio optimization: find best strategy weight combinations.
    Runs top strategies, computes correlation matrix, finds max-Sharpe portfolio.
    """
    from engine import create_strategies

    if coins is None:
        coins = ["BTC"]

    coin = coins[0]
    candles = download_hl_candles(coin, interval, limit)

    if len(candles) < 100:
        return {"error": f"Insufficient data: {len(candles)} candles"}

    strategies = create_strategies()
    equity_curves: Dict[str, List[float]] = {}

    # Run each strategy
    for sw in strategies:
        try:
            sw.initialize(candles)
        except Exception:
            continue

        equity = [10000.0]
        position = None
        entry_price = 0.0

        for i in range(60, len(candles)):
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
                pnl_pct = (price - entry_price) / entry_price
                equity.append(equity[-1] * (1 + pnl_pct))
                position = None
                entry_price = 0.0

        if len(equity) > 10:
            metrics = _calc_metrics(equity, 0, 0, 0)
            equity_curves[sw.state.name] = equity

    if len(equity_curves) < 2:
        return {"error": "Not enough strategies produced equity curves for portfolio optimization"}

    # Pick top_n by final equity
    sorted_strats = sorted(equity_curves.items(), key=lambda x: x[1][-1], reverse=True)[:top_n]
    top_names = [s[0] for s in sorted_strats]
    top_curves = {name: equity_curves[name] for name in top_names}

    # Align curves to same length
    min_len = min(len(c) for c in top_curves.values())
    aligned = {}
    for name, curve in top_curves.items():
        aligned[name] = curve[-min_len:]

    # Compute returns matrix
    returns_matrix = {}
    for name, curve in aligned.items():
        arr = np.array(curve)
        rets = np.diff(arr) / arr[:-1]
        returns_matrix[name] = rets

    # Correlation matrix
    names = list(returns_matrix.keys())
    n = len(names)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(returns_matrix[names[i]], returns_matrix[names[j]])[0, 1]

    # Grid search for optimal weights
    best_sharpe = -999
    best_weights = None
    best_equity = None
    steps = 5  # weight step = 0.2
    weight_values = np.linspace(0, 1, steps + 1)

    for weights in iterproduct(weight_values, repeat=n):
        if abs(sum(weights) - 1.0) > 0.01:
            continue

        # Build portfolio returns
        port_rets = np.zeros(min_len - 1)
        for idx, name in enumerate(names):
            port_rets += weights[idx] * returns_matrix[name]

        if len(port_rets) < 10 or np.std(port_rets) == 0:
            continue

        sharpe = np.mean(port_rets) / np.std(port_rets) * np.sqrt(8760)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = {names[i]: round(weights[i], 2) for i in range(n)}
            # Reconstruct portfolio equity
            port_equity = [10000.0]
            for r in port_rets:
                port_equity.append(port_equity[-1] * (1 + r))
            best_equity = port_equity

    # Walk-forward validation
    wf_results = _walk_forward_validation(aligned, returns_matrix, names, train_ratio=0.7)

    # Diversification ratio
    indiv_vols = [np.std(returns_matrix[name]) for name in names]
    port_vol = np.std(sum(returns_matrix[name] * (1.0 / n) for name in names))
    div_ratio = (np.mean(indiv_vols) / port_vol) if port_vol > 0 else 1.0

    return {
        "timestamp": time.time(),
        "coin": coin,
        "candles": len(candles),
        "strategies_used": top_names,
        "correlation_matrix": {
            "names": names,
            "matrix": corr_matrix.tolist(),
        },
        "optimal_portfolio": {
            "weights": best_weights,
            "sharpe": round(best_sharpe, 3),
            "equity_curve": best_equity[-200:] if best_equity else [],
        },
        "diversification_ratio": round(float(div_ratio), 3),
        "walk_forward": wf_results,
    }


def _walk_forward_validation(
    aligned_curves: Dict[str, List[float]],
    returns_matrix: Dict[str, np.ndarray],
    names: List[str],
    train_ratio: float = 0.7,
) -> dict:
    """Split data into train/test, find optimal weights on train, test on held-out data."""
    n_rets = len(returns_matrix[names[0]])
    split = int(n_rets * train_ratio)

    train_rets = {name: rets[:split] for name, rets in returns_matrix.items()}
    test_rets = {name: rets[split:] for name, rets in returns_matrix.items()}

    # Find best on train
    steps = 5
    weight_values = np.linspace(0, 1, steps + 1)
    best_sharpe = -999
    best_w = None
    n = len(names)

    for weights in iterproduct(weight_values, repeat=n):
        if abs(sum(weights) - 1.0) > 0.01:
            continue
        port_ret = sum(weights[i] * train_rets[names[i]] for i in range(n))
        if np.std(port_ret) == 0:
            continue
        sharpe = np.mean(port_ret) / np.std(port_ret) * np.sqrt(8760)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = weights

    # Evaluate on test
    train_result = {"sharpe": round(best_sharpe, 3), "weights": None}
    test_result = {"sharpe": 0.0}

    if best_w is not None:
        train_result["weights"] = {names[i]: round(best_w[i], 2) for i in range(n)}
        test_port_ret = sum(best_w[i] * test_rets[names[i]] for i in range(n))
        if np.std(test_port_ret) > 0:
            test_result["sharpe"] = round(
                float(np.mean(test_port_ret) / np.std(test_port_ret) * np.sqrt(8760)), 3
            )

    return {"train": train_result, "test": test_result}


def _calc_metrics(
    equity: List[float],
    trades: int,
    wins: int,
    total_pnl: float,
) -> dict:
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
    if trades > 0:
        avg_win = total_pnl / trades if trades > 0 else 0
        profit_factor = max(0, avg_win / (abs(total_pnl / trades) + 0.01))

    return {
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "trades": trades,
        "profit_factor": round(profit_factor, 2),
    }


def get_backtest_results() -> Optional[dict]:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None
