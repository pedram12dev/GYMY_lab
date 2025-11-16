from sqlalchemy.orm import Session
from app.selectors.plan_selectors import ensure_seed_workouts
from app.selectors.plan_selectors import remaining_days_count
from app.selectors.plan_selectors import get_or_create_current_plan
from app.selectors.face_selectors import nearest_by_embedding
from app.modules.face.adapter import image_bytes_to_embedding


def evaluate_login(db: Session, embedding: list[float], accept_threshold: float):
    pair = nearest_by_embedding(db, embedding)
    if not pair: return {"status": "failed", "reason": "no_data"}
    emb, sim = pair
    return {"status": "success" if sim >= accept_threshold else "failed","profile_id": emb.profile_id,"similarity": sim,"accepted": sim >= accept_threshold}

def evaluate_login_image(db: Session, image_bytes: bytes, accept_threshold: float):
    vec = image_bytes_to_embedding(image_bytes)
    return evaluate_login(db, vec.tolist(), accept_threshold)
