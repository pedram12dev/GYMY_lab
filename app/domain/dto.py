from pydantic import BaseModel, Field
from typing import List, Optional

class RegisterResponse(BaseModel):
    member_id: int
    profile_id: int
    embedding_id: int

class LoginSubmitRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    embedding: List[float] = Field(min_length=512, max_length=512)

class PlanWindowResponse(BaseModel):
    plan_id: int
    days: List[dict]  # {day_index, completed, items:[{id, workout_name, sets, reps, duration_seconds, completed}]}

class CompleteRequest(BaseModel):
    day_index: int
    item_ids: Optional[List[int]] = None  # None => complete whole day
