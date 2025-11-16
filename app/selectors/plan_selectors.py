from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.modules.db.models import Workout, WorkoutPlan, PlanDay, PlanItem

def ensure_seed_workouts(db: Session):
    if db.scalar(select(func.count(Workout.id))) == 0:
        seeds = [
            ("Push Ups","strength",None), ("Plank","core",None), ("Squats","strength",None),
            ("Jumping Jacks","cardio",None), ("Lunges","strength",None),
        ]
        for n,c,notes in seeds: db.add(Workout(name=n, category=c, notes=notes))
        db.commit()

def get_or_create_current_plan(db: Session, member_id: int, total_days: int = 30) -> WorkoutPlan:
    plan = db.scalar(select(WorkoutPlan).where(WorkoutPlan.member_id == member_id, WorkoutPlan.completed == False).order_by(WorkoutPlan.created_at.desc()))
    if plan: return plan
    plan = WorkoutPlan(member_id=member_id, total_days=total_days)
    db.add(plan); db.commit(); db.refresh(plan); return plan

def plan_window(db: Session, plan_id: int, start_day: int, window: int) -> List[Dict]:
    days = db.scalars(select(PlanDay).where(PlanDay.plan_id==plan_id, PlanDay.day_index>=start_day, PlanDay.day_index<start_day+window).order_by(PlanDay.day_index)).all()
    result = []
    for d in days:
        items = db.scalars(select(PlanItem).where(PlanItem.plan_day_id==d.id)).all()
        result.append({
            "day_index": d.day_index,
            "completed": d.completed,
            "items": [
                {"id": it.id, "workout_id": it.workout_id, "sets": it.sets, "reps": it.reps,
                 "duration_seconds": it.duration_seconds, "notes": it.notes, "completed": it.completed}
                for it in items
            ],
        })
    return result

def remaining_days_count(db: Session, plan_id: int) -> int:
    return db.scalar(select(func.count(PlanDay.id)).where(PlanDay.plan_id==plan_id, PlanDay.completed==False)) or 0
