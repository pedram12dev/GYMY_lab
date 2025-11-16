import json
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.modules.db.models import Member, Workout, WorkoutPlan, PlanDay, PlanItem
from app.modules.llm.run_pipeline import run as llm_run
from app.modules.llm.schema import WorkoutPlan as LLMPlan

def _allowed_workouts(db: Session) -> List[Dict]:
    return [{"name": w.name} for w in db.scalars(select(Workout)).all()]

def _profile_for_llm(m: Member) -> Dict:
    return {
        "user_id": m.id, "age": 0, "gender": "unknown",
        "goal": m.goal or "general_fitness",
        "fitness_level": m.fitness_level or "beginner",
        "target_zone": "", "free_days": m.free_days or "", "equipment": m.equipment or "",
    }

def persist_llm_days(db: Session, plan: WorkoutPlan, llm_plan: LLMPlan, start_index: Optional[int] = None):
    name_to_id = {w.name: w.id for w in db.scalars(select(Workout)).all()}
    for day in llm_plan.days:
        idx = day.day if start_index is None else start_index
        d = PlanDay(plan_id=plan.id, day_index=idx)
        db.add(d); db.flush()
        for it in day.items:
            wid = name_to_id.get(it.workout_name)
            if not wid: continue
            db.add(PlanItem(
                plan_day_id=d.id, workout_id=wid, sets=it.sets,
                reps=it.reps, duration_seconds=it.duration_seconds, notes=it.notes
            ))
        if start_index is not None: start_index += 1
    db.commit()

def generate_days_if_needed(db: Session, member_id: int, needed_days: int):
    m = db.get(Member, member_id); assert m
    plan = db.scalar(select(WorkoutPlan).where(WorkoutPlan.member_id==member_id, WorkoutPlan.completed==False).order_by(WorkoutPlan.created_at.desc()))
    if not plan: 
        plan = WorkoutPlan(member_id=member_id, total_days=30); db.add(plan); db.commit(); db.refresh(plan)
    existing = db.scalar(select(PlanDay).where(PlanDay.plan_id==plan.id).order_by(PlanDay.day_index).limit(1))
    next_index = (db.scalar(select(PlanDay.day_index).where(PlanDay.plan_id==plan.id).order_by(PlanDay.day_index.desc())) or 0) + 1
    allowed = _allowed_workouts(db)
    llm_json = llm_run(_profile_for_llm(m), allowed, max(needed_days,1))
    llm_plan = LLMPlan.model_validate(llm_json)
    persist_llm_days(db, plan, llm_plan, start_index=next_index)
    return plan

def complete_items_or_day(db: Session, member_id: int, day_index: int, item_ids: Optional[List[int]]):
    plan = db.scalar(select(WorkoutPlan).where(WorkoutPlan.member_id==member_id, WorkoutPlan.completed==False).order_by(WorkoutPlan.created_at.desc()))
    if not plan: raise ValueError("plan not found")
    day = db.scalar(select(PlanDay).where(PlanDay.plan_id==plan.id, PlanDay.day_index==day_index))
    if not day: raise ValueError("day not found")
    now = datetime.utcnow()
    if item_ids:
        items = db.scalars(select(PlanItem).where(PlanItem.plan_day_id==day.id, PlanItem.id.in_(item_ids))).all()
        for it in items:
            it.completed = True; it.completed_at = now
    else:
        # complete whole day
        items = db.scalars(select(PlanItem).where(PlanItem.plan_day_id==day.id)).all()
        for it in items:
            it.completed = True; it.completed_at = now
        day.completed = True; day.completed_at = now
    db.commit()
    # if plan days created < total or remaining < 7 → generate more up to 30 days
    created_days = db.scalar(select(PlanDay).where(PlanDay.plan_id==plan.id).count()) or 0
    done_days = db.scalar(select(PlanDay).where(PlanDay.plan_id==plan.id, PlanDay.completed==True).count()) or 0
    if done_days >= plan.total_days:
        plan.completed = True; db.commit(); return
    remaining_to_create = max(0, plan.total_days - created_days)
    remaining_open = max(0, plan.total_days - done_days)
    if remaining_open > 0 and (remaining_to_create > 0 or (remaining_open < 7 and created_days < plan.total_days)):
        generate_days_if_needed(db, member_id, min(7, remaining_open))
