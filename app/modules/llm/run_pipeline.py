from app.modules.llm.graph import build_graph


def run(profile: dict, workouts: list[dict], horizon_days: int) -> dict:
    g = build_graph()
    out = g.invoke({"profile": profile, "workouts": workouts, "horizon_days": horizon_days})
    return out["plan_validated"]
