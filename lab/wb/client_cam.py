# lab/ws/client_cam.py
import asyncio
import json
import time

import cv2
import websockets


async def run(
    url: str,
    camera_index: int = 0,
    fps: int = 10,
    width: int = 640,
    height: int = 480,
    quality: int = 80,
):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    frame_interval = 1.0 / max(1, fps)

    async with websockets.connect(f"{url}?fps={fps}", max_size=8 * 1024 * 1024) as ws:
        print("[INFO] Connected. Streaming frames...")
        last_send = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed.")
                await asyncio.sleep(0.1)
                continue

            now = time.time()
            if now - last_send >= frame_interval:
                last_send = now
                # encode to JPEG
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not ok:
                    print("[WARN] JPEG encode failed.")
                    continue
                await ws.send(buf.tobytes())

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                print(
                    f"[INFO] latency={data.get('latency_ms')} ms, faces={data.get('faces')}, match={data.get('match')}"
                )
            except asyncio.TimeoutError:
                # no message this cycle; continue
                pass


if __name__ == "__main__":
    asyncio.run(run("ws://localhost:8765/stream", camera_index=0, fps=10))
