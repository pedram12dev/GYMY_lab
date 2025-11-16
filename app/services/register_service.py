import json
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.modules.db.models import FaceEmbedding, Member, Workout
from app.services.shared import get_or_create_member_profile
from app.modules.face.adapter import image_bytes_to_embedding
from app.selectors.plan_selectors import ensure_seed_workouts
from app.services.plan_service import generate_days_if_needed


def register_with_image_and_goal(db: Session, *, phone: str, name: str, goal: str, fitness_level: str|None, free_days: str|None, equipment: str|None, image_bytes: bytes):
    member, profile = get_or_create_member_profile(db, phone=phone, name=name)
    member.goal = goal
    member.fitness_level = fitness_level
    member.free_days = free_days
    member.equipment = equipment
    db.flush()
    # embedding
    emb_vec = image_bytes_to_embedding(image_bytes)
    data = json.dumps([float(x) for x in emb_vec.tolist()])
    db.add(FaceEmbedding(profile_id=profile.id, embedding=data))
    # workouts seed if empty
    ensure_seed_workouts(db)
    db.commit()
    # initial 7-day (or more) generation, then keep generating on progress until 30 days
    generate_days_if_needed(db, member.id, needed_days=7)
    return member.id, profile.id
