"""
HERD Bot — Trading Dashboard Server
FastAPI + WebSocket server on port 8090
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
import requests as http_requests

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from engine import StrategyEngine
from sentiment import HerdSentimentAnalyzer
import bt_engine as bt_module

# ── Globals ───────────────────────────────────────────────────────────────────

engine = StrategyEngine()
sentiment = HerdSentimentAnalyzer()
ws_clients: Set[WebSocket] = set()

# Price data
PRICES = {
    "BTC/USDT": 0.0,
    "ETH/USDT": 0.0,
    "SOL/USDT": 0.0,
    "XRP/USDT": 0.0,
}
PRICE_CHANGES = {
    "BTC/USDT": 0.0,
    "ETH/USDT": 0.0,
    "SOL/USDT": 0.0,
    "XRP/USDT": 0.0,
}
PRICE_HISTORY: Dict[str, list] = {k: [] for k in PRICES}
CANDLE_BUFFER: Dict[str, np.ndarray] = {}
CANDLE_COUNTER = 0
START_TIME = time.time()
BINANCE_WS_CONNECTED = False
backtest_running = False
backtest_progress = 0.0

# ── Fetch initial prices ──────────────────────────────────────────────────────

def fetch_initial_prices():
    """Get current prices from Binance REST API."""
    symbols = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT", "XRP/USDT": "XRPUSDT"}
    for asset, sym in symbols.items():
        try:
            r = http_requests.get(
                f"https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": sym},
                timeout=10,
            )
            if r.status_code == 200:
                d = r.json()
                PRICES[asset] = float(d["lastPrice"])
                PRICE_CHANGES[asset] = float(d["priceChangePercent"])
                PRICE_HISTORY[asset] = [PRICES[asset]] * 50
        except Exception as e:
            print(f"[WARN] Failed to fetch {asset}: {e}")
            PRICES[asset] = random.uniform(50000, 70000) if "BTC" in asset else random.uniform(2000, 4000)


# ── Fetch initial candle data ─────────────────────────────────────────────────

def fetch_initial_candles():
    """Download initial hourly candles for strategy initialization."""
    global CANDLE_BUFFER
    try:
        candles = bt_module.download_klines("BTCUSDT", "1h", 200)
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
    base = 65000
    prices = base + np.cumsum(np.random.randn(n) * 200)
    candles = np.column_stack([
        prices - np.random.rand(n) * 100,
        prices + np.abs(np.random.randn(n)) * 200,
        prices - np.abs(np.random.randn(n)) * 200,
        prices,
        np.random.rand(n) * 1000 + 100,
    ])
    CANDLE_BUFFER["BTC/USDT"] = candles
    engine.initialize_all(candles)


# ── Background: Binance WebSocket ─────────────────────────────────────────────

async def binance_ws_listener():
    """Connect to Binance combined stream WebSocket for real-time prices."""
    import websockets

    global BINANCE_WS_CONNECTED
    streams = "btcusdt@ticker/ethusdt@ticker/solusdt@ticker/xrpusdt@ticker"
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                BINANCE_WS_CONNECTED = True
                print("[Binance WS] Connected ✅")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})

                        symbol = payload.get("s", "")
                        price = float(payload.get("c", 0))
                        change = float(payload.get("P", 0))

                        asset_map = {
                            "BTCUSDT": "BTC/USDT",
                            "ETHUSDT": "ETH/USDT",
                            "SOLUSDT": "SOL/USDT",
                            "XRPUSDT": "XRP/USDT",
                        }
                        asset = asset_map.get(symbol)
                        if asset and price > 0:
                            PRICES[asset] = price
                            PRICE_CHANGES[asset] = change
                            PRICE_HISTORY[asset].append(price)
                            if len(PRICE_HISTORY[asset]) > 200:
                                PRICE_HISTORY[asset] = PRICE_HISTORY[asset][-200:]

                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass

        except Exception as e:
            BINANCE_WS_CONNECTED = False
            print(f"[Binance WS] Disconnected: {e}, reconnecting in 5s...")
            await asyncio.sleep(5)


# ── Background: Strategy tick loop ────────────────────────────────────────────

async def strategy_loop():
    """Periodically process new bars through strategies."""
    global CANDLE_COUNTER
    await asyncio.sleep(10)  # Wait for initial data

    while True:
        try:
            # Simulate a new candle from current prices every 30 seconds
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

                # Run strategies on latest bar
                new_signals = engine.process_tick(buf, len(buf) - 1, PRICES)

                # Update sentiment
                sentiment.update(PRICES, PRICE_CHANGES)

            await asyncio.sleep(30)
        except Exception as e:
            print(f"[Strategy Loop] Error: {e}")
            await asyncio.sleep(10)


# ── Background: WebSocket broadcast ───────────────────────────────────────────

async def ws_broadcast():
    """Broadcast data to all connected dashboard clients."""
    global ws_clients
    global ws_clients
    while True:
        if ws_clients:
            try:
                # Get latest signals
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
                    "binance_connected": BINANCE_WS_CONNECTED,
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
    asyncio.create_task(binance_ws_listener())
    asyncio.create_task(strategy_loop())
    asyncio.create_task(ws_broadcast())
    yield

app = FastAPI(title="HERD Bot", lifespan=lifespan)

# Mount static files
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
        "binance_ws": BINANCE_WS_CONNECTED,
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

    # Run in background thread
    loop = asyncio.get_event_loop()

    def _run():
        global backtest_running, backtest_progress
        try:
            backtest_progress = 0.3
            result = bt_module.run_backtest("BTCUSDT", "1h", 8760)
            backtest_progress = 1.0
            return result
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
            # Keep alive, receive any client messages
            data = await ws.receive_text()
            # Client can send commands
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
