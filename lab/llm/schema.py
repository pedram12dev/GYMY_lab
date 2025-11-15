# lab/llm/schema.py
from typing import List, Optional
from pydantic import BaseModel, Field

class PlanItem(BaseModel):
    workout_name: str
    sets: int = Field(gt=0)
    reps: Optional[int] = None
    duration_seconds: Optional[int] = None
    notes: Optional[str] = None

class PlanDay(BaseModel):
    day: int = Field(gt=0)
    items: List[PlanItem]

class WorkoutPlan(BaseModel):
    profile_id: int
    horizon_days: int = Field(gt=0)
    days: List[PlanDay]
