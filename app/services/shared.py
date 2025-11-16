import math
from typing import Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.modules.db.models import Member, FaceProfile

def normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v*v for v in vec))
    if n == 0: raise ValueError("zero vector")
    return [v/n for v in vec]

def get_or_create_member_profile(db: Session, *, phone: str, name: str) -> Tuple[Member, FaceProfile]:
    m = db.scalar(select(Member).where(Member.phone == phone))
    if not m: m = Member(phone=phone, name=name); db.add(m); db.flush()
    p = db.scalar(select(FaceProfile).where(FaceProfile.member_id == m.id))
    if not p: p = FaceProfile(member_id=m.id); db.add(p); db.flush()
    db.commit(); db.refresh(m); db.refresh(p); return m, p
