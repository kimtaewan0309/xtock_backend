import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Set

import yfinance as yf
from fastapi import WebSocket


class SymbolHub:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.clients: Set[WebSocket] = set()
        self.stream_task: Optional[asyncio.Task] = None
        self.last_price: Optional[float] = None
        self.last_time: Optional[str] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)

        if self.last_price is not None:
            await websocket.send_json({
                "type": "tick",
                "symbol": self.symbol,
                "price": self.last_price,
                "time": self.last_time,
            })

        if self.stream_task is None or self.stream_task.done():
            self.stream_task = asyncio.create_task(self.run_polling())

    def disconnect(self, websocket: WebSocket):
        self.clients.discard(websocket)

    async def broadcast(self, payload: dict):
        dead_clients = []

        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_clients.append(ws)

        for ws in dead_clients:
            self.clients.discard(ws)

    async def run_polling(self):
        ticker = yf.Ticker(self.symbol)

        while self.clients:
            try:
                price = None

                try:
                    fi = ticker.fast_info
                    if fi:
                        price = fi.get("lastPrice") or fi.get("last_price")
                except Exception:
                    pass

                if price is None:
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        price = float(hist["Close"].dropna().iloc[-1])

                if price is not None:
                    now_iso = datetime.now(timezone.utc).astimezone().isoformat()
                    self.last_price = float(price)
                    self.last_time = now_iso

                    await self.broadcast({
                        "type": "tick",
                        "symbol": self.symbol,
                        "price": self.last_price,
                        "time": self.last_time,
                        "source": "polling",
                    })

            except Exception as e:
                await self.broadcast({
                    "type": "error",
                    "symbol": self.symbol,
                    "message": str(e),
                })

            await asyncio.sleep(2)


symbol_hubs: Dict[str, SymbolHub] = {}


def get_hub(symbol: str) -> SymbolHub:
    symbol = symbol.upper()
    if symbol not in symbol_hubs:
        symbol_hubs[symbol] = SymbolHub(symbol)
    return symbol_hubs[symbol]