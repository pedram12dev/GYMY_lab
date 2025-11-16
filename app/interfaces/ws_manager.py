import asyncio
from typing import Dict, Optional
from fastapi import WebSocket

class WSLoginManager:
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._results: Dict[str, dict] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self._connections[session_id] = ws
        self._events[session_id] = asyncio.Event()

    def disconnect(self, session_id: str):
        self._connections.pop(session_id, None)
        self._events.pop(session_id, None)
        self._results.pop(session_id, None)

    def set_result(self, session_id: str, result: dict):
        self._results[session_id] = result
        if ev := self._events.get(session_id): ev.set()

    async def wait_for_result(self, session_id: str, timeout: int) -> Optional[dict]:
        ev = self._events.get(session_id)
        if not ev: return None
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return self._results.get(session_id)
        except asyncio.TimeoutError:
            return None

manager = WSLoginManager()
