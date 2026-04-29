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
            self.stream_task = asyncio.create_task(self.run_stream())

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

    async def run_stream(self):
        """
        Yahoo Finance 실시간 WebSocket 스트림 사용
        """
        try:
            async with yf.AsyncWebSocket(verbose=False) as ws:
                await ws.subscribe(self.symbol)

                async def handle_message(message):
                    parsed = self.parse_message(message)
                    if not parsed:
                        return

                    self.last_price = parsed["price"]
                    self.last_time = parsed["time"]

                    await self.broadcast({
                        "type": "tick",
                        "symbol": self.symbol,
                        "price": self.last_price,
                        "time": self.last_time,
                        "source": "yfinance_stream",
                    })

                # listen()은 message_handler를 받아 실시간 메시지를 처리
                await ws.listen(message_handler=handle_message)

        except Exception as e:
            # 스트림 실패 시 에러 전송
            await self.broadcast({
                "type": "error",
                "symbol": self.symbol,
                "message": str(e),
            })

    def parse_message(self, message) -> Optional[dict]:
        """
        Yahoo/yfinance 메시지는 종목과 필드 구성이 조금 다를 수 있어서
        가격/시간 필드를 방어적으로 추출
        """
        if not isinstance(message, dict):
            return None

        price = (
            message.get("price")
            or message.get("regularMarketPrice")
            or message.get("lastPrice")
            or message.get("last_price")
            or message.get("p")
        )

        if price is None:
            return None

        ts = message.get("time") or message.get("timestamp") or message.get("t")

        if isinstance(ts, (int, float)):
            if ts > 1e12:
                dt_obj = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).astimezone()
            else:
                dt_obj = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
            time_iso = dt_obj.isoformat()
        elif isinstance(ts, str):
            time_iso = ts
        else:
            time_iso = datetime.now(timezone.utc).astimezone().isoformat()

        return {
            "price": float(price),
            "time": time_iso,
        }


symbol_hubs: Dict[str, SymbolHub] = {}


def get_hub(symbol: str) -> SymbolHub:
    symbol = symbol.upper()
    if symbol not in symbol_hubs:
        symbol_hubs[symbol] = SymbolHub(symbol)
    return symbol_hubs[symbol]