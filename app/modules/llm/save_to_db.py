# lab/llm/save_to_db.py
from lab.db.test_database import SessionLocal
from app.models.available_workout import AvailableWorkout
from lab.llm.schema import WorkoutPlan, PlanItem

def persist_plan(plan: WorkoutPlan, profile_id: int):
    db = SessionLocal()
    try:
        for day in plan.days:
            for it in day.items:
                w = db.query(AvailableWorkout).filter_by(name=it.workout_name).first()
                if not w:
                    continue  # یا raise
                # اینجا رکورد WorkoutPlan یا DailyWorkout شما را ایجاد کنید
                # مثال: UserWorkoutPlan(...)
                # db.add(...)
        db.commit()
    finally:
        db.close()
