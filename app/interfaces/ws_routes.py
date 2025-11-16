import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.interfaces.ws_manager import manager
from app.interfaces.deps import get_settings_dep

router = APIRouter()

@router.websocket("/ws/login/{session_id}")
async def login_ws(ws: WebSocket, session_id: str, settings = Depends(get_settings_dep)):
    await manager.connect(session_id, ws)
    try:
        await ws.send_json({"event": "connected", "session_id": session_id, "status": "waiting"})
        while True:
            wait_task = manager.wait_for_result(session_id, timeout=settings.login_timeout_seconds)
            recv_task = ws.receive_text()
            done, _ = await asyncio.wait({wait_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            if wait_task in done:
                result = wait_task.result()
                await ws.send_json(result or {"event": "timeout"})
                break
            if recv_task in done:
                msg = recv_task.result()
                if msg.strip().lower() == "cancel":
                    await ws.send_json({"event": "cancelled"})
                    break
                else:
                    await ws.send_json({"event": "echo", "message": msg})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)
