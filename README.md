# 🐃 HERD — Trading Intelligence Terminal

Live crypto trading dashboard with herd sentiment analysis, 17 algorithmic strategies, and real-time Binance WebSocket prices.

**🔗 Live Dashboard:** http://u08g4gsowo0scos0w4w444w8.89.167.27.180.sslip.io

## Features

### 17 Trading Strategies
| Category | Strategies |
|----------|-----------|
| **Trend** | Aroon Oscillator, EMA Golden Cross, Vortex Indicator, ADX Trend Strength, Choppiness Index, Ehlers Smoother |
| **Mean Reversion** | RSI Divergence, LinReg Channel, Weighted Close Rev, Distance from MA |
| **Breakout** | Ichimoku Cloud, Bollinger Squeeze, BB Width Expansion, N-Bar Breakout |
| **Volume** | OBV Divergence |
| **Momentum** | MACD Momentum |
| **ML** | Anomaly Detection |

### Herd Sentiment Analyzer
- **Fear & Greed Gauge** — Animated semicircle gauge (0-100)
- **Per-Asset Sentiment** — BTC, ETH, SOL, XRP scored -100 to +100
- **Herd Momentum Tracking** — Detects rapid sentiment shifts
- **Whale Activity Alerts** — Simulated large-order detection
- **Contrarian Signals** — Auto-generates SELL when herd is too greedy, BUY when too fearful

### Live Market Data
- Real-time Binance WebSocket (BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT)
- Sparkline price charts per asset
- 24h price change tracking

### Backtest Engine
- Downloads real 1-year hourly data from Binance
- Runs all 17 strategies on historical data
- Ranks by Sharpe ratio with full metrics (return, drawdown, win rate, trades)
- One-click backtest from the dashboard

### Dashboard
- Bloomberg-terminal dark theme
- Real-time WebSocket updates every 2 seconds
- Strategy grid with live PnL, Sharpe, win rate
- Signal feed with BUY/SELL alerts and confidence scores
- Equity curve and comparison charts
- Fully responsive design

## Tech Stack
- **Backend:** Python 3.11, FastAPI, WebSocket, NumPy, Pandas
- **Frontend:** Vanilla HTML/CSS/JS (no frameworks, no build step)
- **Data:** Binance REST + WebSocket APIs
- **Deploy:** Docker, Coolify, Traefik

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server health + uptime |
| GET | `/api/strategies` | All strategies with performance |
| GET | `/api/signals` | Recent trading signals |
| GET | `/api/sentiment` | Herd sentiment + Fear/Greed |
| GET | `/api/performance` | Aggregate performance metrics |
| POST | `/api/strategies/{name}/start` | Start a strategy |
| POST | `/api/strategies/{name}/stop` | Stop a strategy |
| GET | `/api/backtest/results` | Backtest results |
| POST | `/api/backtest/run` | Run backtest |
| WS | `/ws` | Real-time data stream |

## Testing
```bash
python3 -m pytest test_herd_bot.py -v
# 35/35 tests pass
```

## Local Development
```bash
pip install fastapi uvicorn websockets numpy pandas requests
python3 server.py
# Open http://localhost:8090
```

## Repository
https://github.com/exurocommits/herd-bot

---

⚠ **Paper trading only — not financial advice. Never trade money you can't afford to lose.**
