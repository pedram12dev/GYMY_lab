# lab/face/find_nearest.py
"""
Find the most similar face embedding in the lab test DB.
Usage:
    python -m lab.face.find_nearest lab/face/sample.jpg
"""

import sys

import cv2
import numpy as np

from lab.db.models import FaceEmbedding
from lab.db.test_database import SessionLocal
from lab.face.pipeline import (
    cosine_similarity,
    detect_faces,
    get_face_embedding,
    initialize_onnx_sessions,
    sanity_check_embedding,
)


def run(image_path: str):
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

    # 6) Compare with DB embeddings
    db = SessionLocal()
    try:
        records = db.query(FaceEmbedding).all()
        if not records:
            print("[WARN] No embeddings stored in DB.")
            return

        best_id, best_profile, best_sim = None, None, -1.0
        for rec in records:
            emb_back = np.fromstring(rec.embedding, sep=",")
            sim = cosine_similarity(vec, emb_back)
            print(f"[DEBUG] id={rec.id}, profile={rec.profile_id}, sim={sim:.4f}")
            if sim > best_sim:
                best_id, best_profile, best_sim = rec.id, rec.profile_id, sim

        print(
            f"[INFO] Nearest match: id={best_id}, profile_id={best_profile}, similarity={best_sim:.4f}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "lab/face/sample.jpg"
    run(path)
