from typing import Set
from app.modules.llm.schema import WorkoutPlan
def validate_plan(plan_dict: dict, allowed_workouts: Set[str]) -> WorkoutPlan:
    plan = WorkoutPlan.model_validate(plan_dict)
    for d in plan.days:
        for it in d.items:
            if it.workout_name not in allowed_workouts:
                raise ValueError(f"Unknown workout: {it.workout_name}")
            if (it.reps is None) == (it.duration_seconds is None):
                raise ValueError(f"Invalid item spec for {it.workout_name}")
    return plan
