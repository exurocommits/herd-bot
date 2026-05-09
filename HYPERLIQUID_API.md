# HyperLiquid API Reference (extracted from docs + SDK)

## Base URLs
- Mainnet API: `https://api.hyperliquid.xyz`
- Mainnet WS: `wss://api.hyperliquid.xyz/ws`
- Testnet API: `https://api.hyperliquid-testnet.xyz`
- Testnet WS: `wss://api.hyperliquid-testnet.xyz/ws`

## Info Endpoint
All info queries: `POST https://api.hyperliquid.xyz/info`
Content-Type: application/json

### allMids
```json
{"type": "allMids"}
```
Returns: `{"APE": "4.33245", "ARB": "1.21695", "BTC": "103245.0", ...}`

### candleSnapshot (OHLCV)
```json
{
  "type": "candleSnapshot",
  "req": {
    "coin": "BTC",
    "interval": "1h",
    "startTime": 1681923600000,
    "endTime": 1681927200000
  }
}
```
Returns: `[{"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "open_time", "T": "close_time"}, ...]`
Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, 3d, 1w, 1M
Max 5000 candles per request. Paginate using last returned T as next startTime.

### clearinghouseState (user positions)
```json
{"type": "clearinghouseState", "user": "0x..."}
```
Returns: assetPositions, marginSummary, withdrawable, crossMarginSummary

### openOrders
```json
{"type": "openOrders", "user": "0x..."}
```

### userFills / userFillsByTime
```json
{"type": "userFills", "user": "0x..."}
{"type": "userFillsByTime", "user": "0x...", "startTime": 1681923600000}
```
Max 2000 fills per response. 10000 most recent available.

### meta (perp universe)
```json
{"type": "meta"}
```
Returns universe with coin names, szDecimals, etc.

### spotMeta
```json
{"type": "spotMeta"}
```

### fundingHistory
```json
{"type": "fundingHistory", "coin": "BTC", "startTime": 1681923600000}
```

### openInterest
```json
{"type": "openInterest", "coin": "BTC"}
```

## Exchange Endpoint
All exchange actions: `POST https://api.hyperliquid.xyz/exchange`
Requires: action + nonce + signature (signed with eth_account)

### Place Order
```json
{
  "action": {
    "type": "order",
    "orders": [{
      "a": 0,        // asset index
      "b": true,     // isBuy
      "p": "50000.0", // limit price
      "s": "0.01",   // size
      "r": false,    // reduceOnly
      "t": {"limit": {"tif": "Gtc"}}  // Alo, Ioc, Gtc
    }],
    "grouping": "na"
  },
  "nonce": 1681923600000,
  "signature": {...}
}
```

### Cancel Order
```json
{
  "action": {
    "type": "cancel",
    "cancels": [{"a": 0, "o": 91490942}]
  },
  "nonce": 1681923600000,
  "signature": {...}
}
```

## WebSocket
Connect: `wss://api.hyperliquid.xyz/ws`

### Subscribe to allMids
```json
{"method": "subscribe", "subscription": {"type": "allMids"}}
```
Channel: `AllMids` — streams mid price updates for all coins

### Subscribe to L2 Book
```json
{"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}}
```
Channel: `WsBook` — order book updates

### Subscribe to Trades
```json
{"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}}
```
Channel: `trades` — recent trades

### Subscribe to Candle
```json
{"method": "subscribe", "subscription": {"type": "candle", "coin": "BTC", "interval": "1m"}}
```
Channel: `Candle[]` — OHLCV candle updates

### Subscribe to User Events
```json
{"method": "subscribe", "subscription": {"type": "userEvents", "user": "0x..."}}
```

## Python SDK (hyperliquid-python-sdk)
```bash
pip install hyperliquid
```

### Info Client
```python
from hyperliquid.info import Info
info = Info()  # defaults to mainnet
mids = info.all_mids()
candles = info.candles_snapshot("BTC", "1h", start_time, end_time)
state = info.user_state("0x...")
orders = info.open_orders("0x...")
fills = info.user_fills("0x...")
meta = info.meta()
```

### Exchange Client (requires wallet)
```python
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
import eth_account

wallet = eth_account.Account.from_key("0x_PRIVATE_KEY")
exchange = Exchange(wallet, MAINNET_API_URL)
result = exchange.order("BTC", True, 0.01, 50000.0, {"limit": {"tif": "Gtc"}})
```

### WebSocket Manager
```python
from hyperliquid.websocket_manager import WebsocketManager
ws = WebsocketManager("https://api.hyperliquid.xyz")
ws.subscribe("allMids", lambda msg: print(msg))
ws.subscribe("l2Book", lambda msg: print(msg), coin="BTC")
```

## Asset IDs
- Perps: index from meta.universe (0, 1, 2, ...)
- Spot: 10000 + index from spotMeta.universe
- HIP-3: 110000 + offset
- Coin names: "BTC", "ETH", "SOL" for perps; "@{index}" for spot (e.g. "@107" for HYPE)
