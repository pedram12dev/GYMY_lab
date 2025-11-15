# lab/ws/server.py
import asyncio
import json
import time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import websockets

from lab.db.models import FaceEmbedding
from lab.db.test_database import SessionLocal
from lab.face.pipeline import (
    cosine_similarity,
    detect_faces,
    get_face_embedding,
    initialize_onnx_sessions,
    sanity_check_embedding,
)


async def handle_stream(websocket):
    # parse query (fps optional)
    url = urlparse(websocket.request.path)
    params = parse_qs(url.query)
    target_fps = float(params.get("fps", [10])[0])

    det_sess, emb_sess = initialize_onnx_sessions()

    # preload DB embeddings into memory for faster matching
    db = SessionLocal()
    try:
        records = db.query(FaceEmbedding).all()
        gallery = [
            (rec.id, rec.profile_id, np.fromstring(rec.embedding, sep=","))
            for rec in records
        ]
        print(f"[INFO] Loaded {len(gallery)} embeddings into memory.")
    finally:
        db.close()

    frame_interval = 1.0 / max(1.0, target_fps)
    last_time = 0.0

    async for message in websocket:
        # rate limit
        now = time.time()
        if now - last_time < frame_interval:
            continue
        last_time = now

        # decode JPEG bytes to np.ndarray (BGR)
        if isinstance(message, (bytes, bytearray)):
            data = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send(json.dumps({"error": "decode_failed"}))
                continue
        else:
            await websocket.send(json.dumps({"error": "invalid_message_type"}))
            continue

        t0 = time.time()
        boxes = detect_faces(det_sess, frame)
        match_info = None

        if boxes:
            x1, y1, x2, y2 = boxes[0]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                vec = get_face_embedding(emb_sess, crop)
                if sanity_check_embedding(vec) and gallery:
                    # nearest by cosine
                    best = (-1.0, None, None)  # (sim, id, profile_id)
                    for rec_id, prof_id, emb_g in gallery:
                        sim = cosine_similarity(vec, emb_g)
                        if sim > best[0]:
                            best = (sim, rec_id, prof_id)
                    match_info = {
                        "id": best[1],
                        "profile_id": best[2],
                        "similarity": float(best[0]),
                    }

        latency_ms = (time.time() - t0) * 1000.0
        payload = {
            "ts_server": time.time(),
            "latency_ms": round(latency_ms, 2),
            "faces": len(boxes),
            "match": match_info,
        }
        await websocket.send(json.dumps(payload))


async def main():
    async with websockets.serve(
        handle_stream, "0.0.0.0", 8765, max_size=8 * 1024 * 1024
    ):
        print("[INFO] WebSocket server running on ws://localhost:8765/stream")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
