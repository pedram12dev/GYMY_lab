from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, UniqueConstraint, Text, Boolean
from sqlalchemy.orm import relationship
from app.modules.db.test_database import Base

class Member(Base):
    __tablename__ = "members"
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String(32), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    goal = Column(String(64), nullable=True)            # muscle_gain, fat_loss
    fitness_level = Column(String(32), nullable=True)   # beginner, intermediate
    free_days = Column(String(64), nullable=True)       # "friday"
    equipment = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.now)

class FaceProfile(Base):
    __tablename__ = "face_profiles"
    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("face_profiles.id"), nullable=False, index=True)
    embedding = Column(String, nullable=False)  # JSON list[512]
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)
    __table_args__ = (UniqueConstraint("id", name="uq_face_embedding_id"),)


class Workout(Base):
    __tablename__ = "workouts"
    id = Column(Integer, primary_key=True)
    name = Column(String(128), unique=True, index=True, nullable=False)
    category = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)

class WorkoutPlan(Base):
    __tablename__ = "workout_plans"
    id = Column(Integer, primary_key=True)
    member_id = Column(Integer, ForeignKey("members.id"), index=True, nullable=False)
    title = Column(String(128), default="Personal Plan")
    total_days = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.now)
    completed = Column(Boolean, default=False)

class PlanDay(Base):
    __tablename__ = "plan_days"
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("workout_plans.id"), index=True, nullable=False)
    day_index = Column(Integer, nullable=False)  
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
    __table_args__ = (UniqueConstraint("plan_id", "day_index", name="uq_plan_day_idx"),)

class PlanItem(Base):
    __tablename__ = "plan_items"
    id = Column(Integer, primary_key=True)
    plan_day_id = Column(Integer, ForeignKey("plan_days.id"), index=True, nullable=False)
    workout_id = Column(Integer, ForeignKey("workouts.id"), index=True, nullable=False)
    sets = Column(Integer, nullable=False)
    reps = Column(Integer, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    notes = Column(String(255), nullable=True)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
