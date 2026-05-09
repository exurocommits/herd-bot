"""
HERD Bot — Trading Dashboard Server
FastAPI + WebSocket server on port 8090
HyperLiquid as sole data source.
"""
import os
import sys
import time
import asyncio
import json
import random
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from engine import StrategyEngine
from sentiment import HerdSentimentAnalyzer
import bt_engine as bt_module
from hl_client import hl_client, HLWebSocket

# ── Globals ───────────────────────────────────────────────────────────────────

engine = StrategyEngine()
sentiment = HerdSentimentAnalyzer()
ws_clients: Set[WebSocket] = set()

ASSETS = {
    "BTC/USDT": "BTC",
    "ETH/USDT": "ETH",
    "SOL/USDT": "SOL",
    "HYPE/USDT": "HYPE",
}

PRICES = {k: 0.0 for k in ASSETS}
PRICE_CHANGES = {k: 0.0 for k in ASSETS}
PRICE_HISTORY: Dict[str, list] = {k: [] for k in ASSETS}
CANDLE_BUFFER: Dict[str, np.ndarray] = {}
CANDLE_COUNTER = 0
START_TIME = time.time()
HL_WS_CONNECTED = False
backtest_running = False
backtest_progress = 0.0
PORTFOLIO_RESULT = None

# HL market data
FUNDING_RATES: Dict[str, dict] = {}
OPEN_INTEREST: Dict[str, dict] = {}
ORDER_BOOKS: Dict[str, dict] = {}


# ── Fetch initial prices ──────────────────────────────────────────────────────

def fetch_initial_prices():
    """Get current prices from HyperLiquid REST API."""
    global FUNDING_RATES, OPEN_INTEREST
    try:
        mids = hl_client.fetch_all_mids()
        for asset, coin in ASSETS.items():
            price = mids.get(coin, 0)
            if price > 0:
                prev = PRICES.get(asset, 0)
                PRICES[asset] = price
                if prev > 0:
                    PRICE_CHANGES[asset] = ((price / prev) - 1) * 100
                PRICE_HISTORY[asset].append(price)
                if len(PRICE_HISTORY[asset]) > 200:
                    PRICE_HISTORY[asset] = PRICE_HISTORY[asset][-200:]

        print(f"[Init] Prices loaded: BTC={PRICES['BTC/USDT']:.2f}")

    except Exception as e:
        print(f"[WARN] Failed to fetch HL prices: {e}")
        # Fallback defaults
        defaults = {"BTC/USDT": 65000, "ETH/USDT": 3500, "SOL/USDT": 150, "HYPE/USDT": 25}
        for k, v in defaults.items():
            PRICES[k] = v
            PRICE_HISTORY[k] = [v] * 50

    # Fetch funding + OI for top coins
    try:
        for coin in ["BTC", "ETH", "SOL", "HYPE"]:
            try:
                fh = hl_client.fetch_funding_history(coin, int((time.time() - 86400) * 1000))
                if fh:
                    latest = fh[-1] if isinstance(fh, list) else fh
                    FUNDING_RATES[coin] = {
                        "rate": float(latest.get("fundingRate", 0)),
                        "timestamp": latest.get("time", 0),
                    }
            except Exception:
                FUNDING_RATES[coin] = {"rate": 0.0, "timestamp": 0}

            try:
                oi = hl_client.fetch_open_interest(coin)
                OPEN_INTEREST[coin] = oi
            except Exception:
                OPEN_INTEREST[coin] = {}
    except Exception as e:
        print(f"[WARN] Funding/OI fetch failed: {e}")


def fetch_initial_candles():
    """Download initial hourly candles for strategy initialization."""
    global CANDLE_BUFFER
    try:
        candles = bt_module.download_hl_candles("BTC", "1h", 200)
        if len(candles) > 50:
            CANDLE_BUFFER["BTC/USDT"] = candles
            engine.initialize_all(candles)
            print(f"[Init] Loaded {len(candles)} candles, {len(engine.strategies)} strategies ready")
        else:
            print("[Init] Not enough candle data, using synthetic")
            _use_synthetic_candles()
    except Exception as e:
        print(f"[Init] Candle download failed: {e}, using synthetic")
        _use_synthetic_candles()


def _use_synthetic_candles():
    n = 200
    base = PRICES.get("BTC/USDT", 65000)
    prices = base + np.cumsum(np.random.randn(n) * base * 0.003)
    candles = np.column_stack([
        prices - np.random.rand(n) * base * 0.002,
        prices + np.abs(np.random.randn(n)) * base * 0.003,
        prices - np.abs(np.random.randn(n)) * base * 0.003,
        prices,
        np.random.rand(n) * 1000 + 100,
    ])
    CANDLE_BUFFER["BTC/USDT"] = candles
    engine.initialize_all(candles)


# ── Background: HL WebSocket ─────────────────────────────────────────────────

async def hl_ws_listener():
    """Connect to HyperLiquid WebSocket for live prices."""
    global HL_WS_CONNECTED
    hl_ws = HLWebSocket()

    async def on_all_mids(data):
        global HL_WS_CONNECTED
        HL_WS_CONNECTED = True
        mids_data = data.get("data", {})
        for asset, coin in ASSETS.items():
            price = float(mids_data.get(coin, 0))
            if price > 0:
                prev = PRICES.get(asset, 0)
                PRICES[asset] = price
                if prev > 0:
                    PRICE_CHANGES[asset] = ((price / prev) - 1) * 100
                PRICE_HISTORY[asset].append(price)
                if len(PRICE_HISTORY[asset]) > 200:
                    PRICE_HISTORY[asset] = PRICE_HISTORY[asset][-200:]

    hl_ws.subscribe({"type": "allMids"}, on_all_mids)
    await hl_ws.connect()


# ── Background: Strategy tick loop ────────────────────────────────────────────

async def strategy_loop():
    """Periodically process new bars through strategies."""
    global CANDLE_COUNTER
    await asyncio.sleep(10)

    while True:
        try:
            CANDLE_COUNTER += 1
            price = PRICES.get("BTC/USDT", 65000)
            if price > 0 and "BTC/USDT" in CANDLE_BUFFER:
                buf = CANDLE_BUFFER["BTC/USDT"]
                noise = price * 0.001
                new_candle = np.array([[
                    price + random.gauss(0, noise),
                    price + abs(random.gauss(0, noise)),
                    price - abs(random.gauss(0, noise)),
                    price,
                    random.uniform(100, 2000),
                ]])
                buf = np.vstack([buf, new_candle])
                if len(buf) > 300:
                    buf = buf[-300:]
                CANDLE_BUFFER["BTC/USDT"] = buf

                new_signals = engine.process_tick(buf, len(buf) - 1, PRICES)
                sentiment.update(PRICES, PRICE_CHANGES)

                # Update funding/OI every 10 ticks
                if CANDLE_COUNTER % 10 == 0:
                    _refresh_market_data()

            await asyncio.sleep(30)
        except Exception as e:
            print(f"[Strategy Loop] Error: {e}")
            await asyncio.sleep(10)


def _refresh_market_data():
    """Refresh funding rates and OI data."""
    global FUNDING_RATES, OPEN_INTEREST
    for coin in ["BTC", "ETH", "SOL", "HYPE"]:
        try:
            fh = hl_client.fetch_funding_history(coin, int((time.time() - 86400) * 1000))
            if fh and isinstance(fh, list) and len(fh) > 0:
                latest = fh[-1]
                FUNDING_RATES[coin] = {
                    "rate": float(latest.get("fundingRate", 0)),
                    "timestamp": latest.get("time", 0),
                }
        except Exception:
            pass

        try:
            OPEN_INTEREST[coin] = hl_client.fetch_open_interest(coin)
        except Exception:
            pass


# ── Background: WebSocket broadcast ───────────────────────────────────────────

async def ws_broadcast():
    """Broadcast data to all connected dashboard clients."""
    while True:
        if ws_clients:
            try:
                recent_signals = engine.signal_feed[-10:]
                sent_data = sentiment.to_dict()

                payload = {
                    "type": "tick",
                    "prices": PRICES,
                    "price_changes": PRICE_CHANGES,
                    "price_history": {k: v[-60:] for k, v in PRICE_HISTORY.items()},
                    "sentiment": sent_data,
                    "signals": recent_signals,
                    "strategies": engine.get_strategies_summary(),
                    "performance": engine.get_performance(),
                    "hl_connected": HL_WS_CONNECTED,
                    "funding_rates": FUNDING_RATES,
                    "open_interest": OPEN_INTEREST,
                    "uptime": round(time.time() - START_TIME),
                    "timestamp": time.time(),
                }

                msg = json.dumps(payload)
                disconnected = set()
                for ws in list(ws_clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        disconnected.add(ws)
                ws_clients.difference_update(disconnected)

            except Exception as e:
                print(f"[Broadcast] Error: {e}")

        await asyncio.sleep(2)


# ── FastAPI setup ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    fetch_initial_prices()
    fetch_initial_candles()
    asyncio.create_task(hl_ws_listener())
    asyncio.create_task(strategy_loop())
    asyncio.create_task(ws_broadcast())
    yield

app = FastAPI(title="HERD Bot", lifespan=lifespan)

static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(static_path, "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "uptime": round(time.time() - START_TIME),
        "strategies": len(engine.strategies),
        "hl_ws": HL_WS_CONNECTED,
        "ws_clients": len(ws_clients),
    }


@app.get("/api/strategies")
async def get_strategies():
    return engine.get_strategies_summary()


@app.post("/api/strategies/{name}/start")
async def start_strategy(name: str):
    ok = engine.start_strategy(name)
    return {"success": ok, "name": name}


@app.post("/api/strategies/{name}/stop")
async def stop_strategy(name: str):
    ok = engine.stop_strategy(name)
    return {"success": ok, "name": name}


@app.get("/api/signals")
async def get_signals():
    return engine.signal_feed[-50:]


@app.get("/api/sentiment")
async def get_sentiment():
    return sentiment.to_dict()


@app.get("/api/performance")
async def get_performance():
    return engine.get_performance()


@app.get("/api/funding")
async def get_funding():
    return FUNDING_RATES


@app.get("/api/open-interest")
async def get_open_interest():
    return OPEN_INTEREST


@app.get("/api/book/{coin}")
async def get_book(coin: str):
    coin = coin.upper()
    try:
        book = hl_client.fetch_l2_book(coin)
        return book
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/backtest/results")
async def backtest_results():
    results = bt_module.get_backtest_results()
    return results or {"results": [], "message": "No backtest results yet. Run POST /api/backtest/run first."}


@app.post("/api/backtest/run")
async def run_backtest():
    global backtest_running, backtest_progress
    if backtest_running:
        return {"status": "already_running", "progress": backtest_progress}

    backtest_running = True
    backtest_progress = 0.0

    loop = asyncio.get_event_loop()

    def _run():
        global backtest_running, backtest_progress
        try:
            backtest_progress = 0.3
            result = bt_module.run_backtest("BTC", "1h", 8760)
            backtest_progress = 1.0
            return result
        finally:
            backtest_running = False

    result = await loop.run_in_executor(None, _run)
    return result


@app.get("/api/portfolio/results")
async def portfolio_results():
    global PORTFOLIO_RESULT
    if PORTFOLIO_RESULT:
        return PORTFOLIO_RESULT
    return {"message": "No portfolio results yet. Run POST /api/portfolio/run first."}


@app.post("/api/portfolio/run")
async def run_portfolio():
    global backtest_running, PORTFOLIO_RESULT
    if backtest_running:
        return {"status": "already_running"}

    backtest_running = True
    loop = asyncio.get_event_loop()

    def _run():
        global backtest_running, PORTFOLIO_RESULT
        try:
            PORTFOLIO_RESULT = bt_module.run_portfolio_backtest(["BTC"], "1h", 4380, 5)
            return PORTFOLIO_RESULT
        finally:
            backtest_running = False

    result = await loop.run_in_executor(None, _run)
    return result


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global ws_clients
    await ws.accept()
    ws_clients.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_clients.discard(ws)
    except Exception:
        ws_clients.discard(ws)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
