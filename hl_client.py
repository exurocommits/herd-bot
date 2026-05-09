"""
HyperLiquid data client — handles REST + WebSocket connections.
No SDK dependency — uses raw requests/websockets for minimal deps.
"""
import json
import time
import asyncio
import numpy as np
import requests
from typing import Dict, List, Optional, Callable

BASE_URL = "https://api.hyperliquid.xyz"
WS_URL = "wss://api.hyperliquid.xyz/ws"
REQUEST_TIMEOUT = 15


class HyperLiquidClient:
    """REST client for HyperLiquid API."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self._session = requests.Session()
        self._meta_cache: Optional[dict] = None
        self._meta_ts: float = 0

    def _post(self, payload: dict) -> any:
        resp = self._session.post(
            f"{self.base_url}/info",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Market Data ──────────────────────────────────────────────────────────

    def fetch_all_mids(self) -> Dict[str, float]:
        """Get mid prices for all coins."""
        data = self._post({"type": "allMids"})
        return {k: float(v) for k, v in data.items()}

    def fetch_meta(self) -> dict:
        """Get perp universe metadata (cached for 60s)."""
        now = time.time()
        if self._meta_cache and (now - self._meta_ts) < 60:
            return self._meta_cache
        self._meta_cache = self._post({"type": "meta"})
        self._meta_ts = now
        return self._meta_cache

    def fetch_candle_snapshot(
        self,
        coin: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fetch OHLCV candle data. Returns numpy array shape (N, 5): [open, high, low, close, volume].
        Paginates automatically (max 5000 per request).
        """
        all_rows: List[List[float]] = []
        cursor = start_time
        max_per_req = 5000

        while True:
            req: dict = {"coin": coin, "interval": interval}
            if cursor is not None:
                req["startTime"] = cursor
            if end_time is not None:
                req["endTime"] = end_time

            data = self._post({"type": "candleSnapshot", "req": req})

            if not data:
                break

            batch = []
            for c in data:
                try:
                    batch.append([
                        float(c["o"]),
                        float(c["h"]),
                        float(c["l"]),
                        float(c["c"]),
                        float(c["v"]),
                    ])
                except (KeyError, ValueError, TypeError):
                    continue

            all_rows.extend(batch)

            if len(batch) < max_per_req:
                break

            # Next page starts after last candle's close time
            last_close = data[-1].get("T") or data[-1].get("t")
            if last_close is None:
                break
            cursor = int(last_close) + 1

            if end_time is not None and cursor >= end_time:
                break

        if not all_rows:
            return np.empty((0, 5), dtype=float)

        # Oldest first
        return np.array(all_rows, dtype=float)

    def fetch_funding_history(self, coin: str, start_time: int) -> List[dict]:
        """Fetch funding rate history."""
        data = self._post({"type": "fundingHistory", "coin": coin, "startTime": start_time})
        return data if isinstance(data, list) else []

    def fetch_open_interest(self, coin: str) -> dict:
        """Fetch current open interest for a coin."""
        return self._post({"type": "openInterest", "coin": coin})

    def fetch_l2_book(self, coin: str) -> dict:
        """Fetch L2 order book snapshot."""
        return self._post({"type": "l2Book", "coin": coin})

    def fetch_recent_trades(self, coin: str) -> list:
        """Fetch recent trades (via WS subscription data or recent fills)."""
        # Not a direct REST endpoint for public trades; use WS instead
        # For now, return empty — real implementation uses WS
        return []


# ── WebSocket Manager ────────────────────────────────────────────────────────

class HLWebSocket:
    """Async WebSocket client for HyperLiquid live data."""

    def __init__(self, ws_url: str = WS_URL):
        self.ws_url = ws_url
        self._ws = None
        self._subscriptions: Dict[str, Callable] = {}
        self._running = False
        self._reconnect_delay = 1.0

    async def connect(self):
        """Connect and start message loop with reconnection."""
        import websockets

        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=20, ping_timeout=10
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    print("[HL WS] Connected ✅")

                    # Re-subscribe
                    for sub_key, callback in list(self._subscriptions.items()):
                        sub_msg = json.loads(sub_key)
                        await ws.send(json.dumps({"method": "subscribe", "subscription": sub_msg}))

                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            channel = data.get("channel", "")
                            # Find matching callback
                            for sub_key, callback in list(self._subscriptions.items()):
                                sub_msg = json.loads(sub_key)
                                sub_type = sub_msg.get("type", "")
                                if channel == "allMids" and sub_type == "allMids":
                                    await callback(data)
                                elif channel == "l2Book" and sub_type == "l2Book":
                                    await callback(data)
                                elif channel == "trades" and sub_type == "trades":
                                    await callback(data)
                                elif "Candle" in channel and sub_type == "candle":
                                    await callback(data)
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                self._ws = None
                print(f"[HL WS] Disconnected: {e}, reconnecting in {self._reconnect_delay:.0f}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()

    def subscribe(self, subscription: dict, callback: Callable):
        """Register a subscription. callback = async fn(data)."""
        key = json.dumps(subscription, sort_keys=True)
        self._subscriptions[key] = callback

    def unsubscribe(self, subscription: dict):
        key = json.dumps(subscription, sort_keys=True)
        self._subscriptions.pop(key, None)


# ── Singleton ────────────────────────────────────────────────────────────────

hl_client = HyperLiquidClient()
