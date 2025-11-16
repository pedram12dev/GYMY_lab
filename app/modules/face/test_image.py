# lab/face/test_db.py
"""
Generate a face embedding from an image and store it in the lab test DB.
Usage:
    python -m lab.face.test_db lab/face/sample.jpg
"""

import sys

import cv2
import numpy as np

from lab.db.models import FaceEmbedding

# ✅ اتصال به دیتابیس آزمایشگاه
from lab.db.test_database import SessionLocal
from lab.face.pipeline import (
    detect_faces,
    get_face_embedding,
    initialize_onnx_sessions,
    sanity_check_embedding,
)


def run(image_path: str, profile_id: int = 1, confidence: float = 1.0):
    # 1) Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 2) Init models
    det_session, emb_session = initialize_onnx_sessions()

    # 3) Detect faces
    boxes = detect_faces(det_session, img)
    print(f"[INFO] Detected {len(boxes)} face(s).")
    if not boxes:
        print("[WARN] No face detected.")
        return

    # 4) Crop first face
    x1, y1, x2, y2 = boxes[0]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        print("[ERROR] Cropped face is empty.")
        return

    # 5) Get embedding
    vec = get_face_embedding(emb_session, crop)
    if not sanity_check_embedding(vec):
        print("[ERROR] Invalid embedding.")
        return

    # 6) Convert to CSV string
    emb_str = ",".join(map(str, vec.tolist()))

    # 7) Store in lab DB
    db = SessionLocal()
    try:
        record = FaceEmbedding(
            profile_id=profile_id, embedding=emb_str, confidence=confidence
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        print(
            f"[INFO] Saved FaceEmbedding id={record.id}, profile_id={profile_id}, len=512"
        )

        # 8) Read back and verify
        loaded = db.query(FaceEmbedding).filter_by(id=record.id).first()
        emb_back = np.fromstring(loaded.embedding, sep=",")
        same_len = emb_back.shape[0] == vec.shape[0]
        print(f"[INFO] Loaded len={emb_back.shape[0]} (same_len={same_len})")
        print(f"[INFO] First 5 values (back): {emb_back[:5]}")
    finally:
        db.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "lab/face/sample.jpg"
    run(path, profile_id=1, confidence=1.0)
