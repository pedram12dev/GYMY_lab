import json, math
from typing import List, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.modules.db.models import FaceEmbedding

def _normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(v*v for v in vec))
    if n == 0: raise ValueError("zero vector")
    return [v/n for v in vec]

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def nearest_by_embedding(db: Session, vec: List[float]) -> Optional[Tuple[FaceEmbedding, float]]:
    rows = db.execute(select(FaceEmbedding)).scalars().all()
    if not rows: return None
    v = np.array(_normalize(vec), dtype=np.float32)
    best, best_sim = None, -2.0
    for r in rows:
        arr = np.array(json.loads(r.embedding), dtype=np.float32)
        sim = _cosine(v, arr)
        if sim > best_sim:
            best, best_sim = r, sim
    return (best, best_sim) if best else None
