# lab/llm/validate.py
from typing import Set
from lab.llm.schema import WorkoutPlan, PlanItem

def validate_plan(plan_dict: dict, allowed_workouts: Set[str]) -> WorkoutPlan:
    plan = WorkoutPlan.model_validate(plan_dict)
    for d in plan.days:
        for it in d.items:
            if it.workout_name not in allowed_workouts:
                raise ValueError(f"Unknown workout: {it.workout_name}")
            # exactly one of reps or duration_seconds must be set
            if (it.reps is None) == (it.duration_seconds is None):
                raise ValueError(f"Invalid item spec for {it.workout_name}")
    return plan
