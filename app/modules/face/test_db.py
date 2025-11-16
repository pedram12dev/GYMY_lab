# lab/face/test_db.py
"""
Test script: generate face embedding from image and store it in DB.
Usage:
    python -m lab.face.test_db lab/face/sample.jpg
"""

import sys

import cv2
import numpy as np

from lab.db.models import FaceEmbedding

# --- Import DB models and session ---
from lab.db.test_database import SessionLocal
from lab.face.pipeline import (
    detect_faces,
    get_face_embedding,
    initialize_onnx_sessions,
    sanity_check_embedding,
)


def run(image_path: str, profile_id: int = 1):
    # --- Load image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # --- Initialize models ---
    det_session, emb_session = initialize_onnx_sessions()

    # --- Detect faces ---
    boxes = detect_faces(det_session, img)
    print(f"[INFO] Detected {len(boxes)} face(s).")

    if not boxes:
        print("[WARN] No face detected.")
        return

    # --- Process first face ---
    x1, y1, x2, y2 = boxes[0]
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print("[ERROR] Cropped face is empty.")
        return

    # --- Extract embedding ---
    vec = get_face_embedding(emb_session, crop)
    ok = sanity_check_embedding(vec)

    if not ok:
        print("[ERROR] Invalid embedding.")
        return

    # --- Convert embedding to CSV string ---
    emb_str = ",".join(map(str, vec.tolist()))

    # --- Store in DB ---
    db = SessionLocal()
    face_emb = FaceEmbedding(profile_id=profile_id, embedding=emb_str, confidence=1.0)
    db.add(face_emb)
    db.commit()
    db.refresh(face_emb)

    print(f"[INFO] Saved embedding with id={face_emb.id}, profile_id={profile_id}")

    # --- Read back and check ---
    saved = db.query(FaceEmbedding).filter_by(id=face_emb.id).first()
    emb_back = np.fromstring(saved.embedding, sep=",")
    print(f"[INFO] Loaded embedding length: {emb_back.shape[0]}")
    print(f"[INFO] First 5 values (back): {emb_back[:5]}")

    db.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "lab/face/sample.jpg"
    run(path, profile_id=1)
