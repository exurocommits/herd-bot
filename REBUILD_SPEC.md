# HERD Bot v2 — HyperLiquid Rebuild Spec

## Mission
Rebuild herd-bot to use HyperLiquid as the primary (and only) data source. Remove ALL Binance references.
Focus: **strategy testing + portfolio optimization for high profit returns**.

## What EXISTS (keep + modify)
The project is at `/home/node/.openclaw/workspace/herd-bot/`. Files:
- `engine.py` (520 lines) — StrategyEngine + 17 strategy wrappers. KEEP as-is (strategies are data-source agnostic, they just take OHLCV numpy arrays)
- `sentiment.py` (160 lines) — HerdSentimentAnalyzer. KEEP but enhance with HL-specific data (funding rate, OI)
- `static/index.html` (785 lines) — Bloomberg dark dashboard. KEEP but update for HL features
- `bt_engine.py` (190 lines) — Backtest engine. REWRITE data source to HL candleSnapshot
- `server.py` (365 lines) — FastAPI server. REWRITE data feeds to HL
- `test_herd_bot.py` (344 lines) — Tests. UPDATE for new HL interfaces
- `xrpl_bot/` — Strategy source files. KEEP (read-only)
- `Dockerfile` — KEEP

## HyperLiquid API Reference
See `HYPERLIQUID_API.md` in this directory — full API docs with endpoints, WebSocket, Python SDK examples.

## Changes Required

### 1. NEW FILE: `hl_client.py` — HyperLiquid Data Client
```python
"""
HyperLiquid data client — handles REST + WebSocket connections.
No SDK dependency — uses raw requests/websockets for minimal deps.
"""
```
Must implement:
- `fetch_all_mids() -> dict` — POST to `/info` with `{"type": "allMids"}`
- `fetch_candle_snapshot(coin, interval, start_time, end_time) -> np.ndarray` — returns OHLCV array [open, high, low, close, volume]
- `fetch_funding_history(coin, start_time) -> list` — funding rate data
- `fetch_open_interest(coin) -> dict` — current OI
- `fetch_meta() -> dict` — perp universe metadata
- `fetch_l2_book(coin) -> dict` — order book snapshot
- `ws_connect()` — async WebSocket to `wss://api.hyperliquid.xyz/ws`
- `ws_subscribe_all_mids(callback)` — subscribe to live prices
- `ws_subscribe_candle(coin, interval, callback)` — subscribe to candle updates
- `ws_subscribe_trades(coin, callback)` — subscribe to trade feed
- Reconnection logic with exponential backoff

### 2. REWRITE: `server.py` — Replace Binance with HyperLiquid
- Remove ALL Binance references (REST ticker, WS streams, symbol mapping)
- Use `hl_client.py` for all data
- Price feed: HL `allMids` WebSocket → stream BTC, ETH, SOL, HYPE prices
- Add HL-specific API endpoints:
  - `GET /api/funding` — funding rates for top coins
  - `GET /api/open-interest` — OI data
  - `GET /api/book/{coin}` — L2 order book
- Keep all existing endpoints (/api/strategies, /api/signals, /api/sentiment, etc.)
- Keep FastAPI + WebSocket broadcast to dashboard clients

### 3. REWRITE: `bt_engine.py` — Backtest with HL Candles
- Replace `download_klines()` with `download_hl_candles()` using HL candleSnapshot
- Paginate properly (max 5000 candles per request)
- Support multiple coins: BTC, ETH, SOL, HYPE
- Add **portfolio backtesting**: run multiple strategies with allocation weights
- Portfolio metrics: portfolio Sharpe, Sortino, max DD, correlation matrix, diversification ratio
- Add **strategy combination optimizer**: find best weight combinations across top strategies
- Add **walk-forward validation**: split data into train/test periods

### 4. UPDATE: `sentiment.py` — Add HL Market Signals
- Keep existing Fear & Greed + sentiment scoring
- Add **funding rate signal**: extreme positive funding = crowd is long (contrarian SELL), extreme negative = crowd is short (contrarian BUY)
- Add **OI momentum**: rapid OI increase + price up = strong trend; OI increase + price down = potential squeeze
- Add **liquidation cascade detection**: rapid price move + OI drop = liquidation event
- Add **order book imbalance**: bid/ask depth ratio as sentiment proxy

### 5. UPDATE: `static/index.html` — Dashboard Enhancements
Add these panels to the existing Bloomberg dark dashboard:
- **Funding Rate Panel**: live funding rates for each coin, bar chart, extreme highlighting
- **Open Interest Tracker**: OI trends, OI vs price overlay chart
- **Portfolio Optimizer Panel**: 
  - Select strategies + allocation weights
  - Run portfolio backtest
  - Show efficient frontier chart
  - Show optimal weights (max Sharpe portfolio)
- **Strategy Comparison Heatmap**: color-coded grid of strategy metrics
- **Walk-Forward Results**: train/test split performance bars
- Keep existing: price cards, strategy grid, signal feed, sentiment gauge, equity curve
- Update assets: BTC, ETH, SOL, HYPE (replace XRP with HYPE)

### 6. UPDATE: `test_herd_bot.py`
- Test `hl_client.py` functions (mock HTTP responses)
- Test new bt_engine with mock HL candle data
- Test funding rate + OI sentiment signals
- Test portfolio optimization
- Keep existing engine + strategy tests

## Dependencies
Keep existing: fastapi, uvicorn, websockets, numpy, pandas, requests
DO NOT add: hyperliquid SDK (use raw API calls), eth_account (no trading yet)

## Design Principles
- Pure Python, no C++ deps
- Single HTML file dashboard (no frameworks, no build step)
- Bloomberg terminal quality UI (dark theme, neon green #00ff41 accent)
- Real HL data for everything (prices, candles, funding, OI)
- Paper trading only (no real order execution yet — that's v3)
- 17 existing strategies from xrpl_bot MUST keep working
- Portfolio optimization = find the best combination of strategies + weights for max risk-adjusted return

## File sizes target
- `hl_client.py`: ~200 lines
- `server.py`: ~400 lines
- `bt_engine.py`: ~350 lines (with portfolio optimizer)
- `sentiment.py`: ~250 lines
- `static/index.html`: ~1200 lines (with new panels)
- `test_herd_bot.py`: ~400 lines

## Test & Deploy
After writing all files:
1. Run: `cd /home/node/.openclaw/workspace/herd-bot && python3 -m pytest test_herd_bot.py -v`
2. Fix any import or runtime errors
3. Start server: `python3 server.py` (port 8090)
4. Verify: `curl http://localhost:8090/api/health` and `curl http://localhost:8090/api/strategies`
