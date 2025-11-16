from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.interfaces.deps import get_db, get_settings_dep
from app.domain.dto import RegisterResponse, LoginSubmitRequest, PlanWindowResponse, CompleteRequest
from app.services.register_service import register_with_image_and_goal
from app.services.login_service import evaluate_login, evaluate_login_image
from app.interfaces.ws_manager import manager
from app.modules.db.models import Member, WorkoutPlan, PlanDay, PlanItem, Workout
from app.selectors.plan_selectors import get_or_create_current_plan, plan_window, remaining_days_count, ensure_seed_workouts
from app.services.plan_service import generate_days_if_needed, complete_items_or_day

router = APIRouter()

@router.post("/register-image", response_model=RegisterResponse)
async def register_image(
    phone: str = Form(...),
    name: str = Form(...),
    goal: str = Form(...),
    fitness_level: str = Form("beginner"),
    free_days: str = Form(""),
    equipment: str = Form(""),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    data = await image.read()
    mid, pid = register_with_image_and_goal(
        db, phone=phone, name=name, goal=goal,
        fitness_level=fitness_level, free_days=free_days, equipment=equipment,
        image_bytes=data
    )
    return RegisterResponse(member_id=mid, profile_id=pid, embedding_id=0)

@router.post("/login/submit")
def login_submit(payload: LoginSubmitRequest, db: Session = Depends(get_db), settings = Depends(get_settings_dep)):
    result = evaluate_login(db, payload.embedding, accept_threshold=settings.cosine_login_threshold)
    manager.set_result(payload.session_id, {"event": "result", **result})
    return {"queued": True}

@router.post("/login/image")
async def login_image(
    session_id: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings = Depends(get_settings_dep),
):
    data = await image.read()
    result = evaluate_login_image(db, data, accept_threshold=settings.cosine_login_threshold)
    manager.set_result(session_id, {"event": "result", **result})
    return {"queued": True}

@router.get("/profile/{member_id}/plan", response_model=PlanWindowResponse)
def profile_plan(member_id: int, window: int = 7, start_day: int = 1, db: Session = Depends(get_db)):
    ensure_seed_workouts(db)
    plan = get_or_create_current_plan(db, member_id)
    # if not enough created days, generate
    created_last = db.scalar(select(PlanDay.day_index).where(PlanDay.plan_id==plan.id).order_by(PlanDay.day_index.desc()))
    current_created = created_last or 0
    need_create = max(0, (start_day + window - 1) - current_created)
    if need_create > 0:
        generate_days_if_needed(db, member_id, need_create)
    days = plan_window(db, plan.id, start_day, window)
    return PlanWindowResponse(plan_id=plan.id, days=days)

@router.post("/profile/{member_id}/plan/complete")
def plan_complete(member_id: int, payload: CompleteRequest, db: Session = Depends(get_db)):
    complete_items_or_day(db, member_id, payload.day_index, payload.item_ids)
    return {"status": "ok"}
