# lab/llm/run_pipeline.py
import json
from lab.llm.graph import build_graph

def mock_profile():
    return {
        "id": 1,
        "user_id": 1,
        "age": 28,
        "gender": "male",
        "weight": 75.0,
        "height": 178.0,
        "goal": "muscle_gain",
        "fitness_level": "beginner",
        "target_zone": "abs,arms,legs",
        "free_days": "friday",
        "equipment": "bodyweight,dumbbell"
    }

def mock_available_workouts():
    return [
        {"id": 1, "name": "Push-up", "goal": "strength", "body_area": "arms,chest"},
        {"id": 2, "name": "Plank", "goal": "general_fitness", "body_area": "abs,back"},
        {"id": 3, "name": "Squat", "goal": "strength", "body_area": "legs,glutes"},
        {"id": 4, "name": "Jumping Jacks", "goal": "fat_loss", "body_area": "full body"},
        {"id": 5, "name": "Yoga Stretch", "goal": "flexibility", "body_area": "full body"}
    ]

def main(horizon_days: int = 7):
    state = {
        "profile": mock_profile(),
        "workouts": mock_available_workouts(),
        "horizon_days": horizon_days
    }
    graph = build_graph()
    result = graph.invoke(state)
    plan = result["plan_validated"]

    print("[INFO] Plan JSON (validated):")
    print(json.dumps(plan, indent=2))

    # pretty print view
    print("\n[VIEW]")
    print(f"User profile_id={plan['profile_id']} | horizon={plan['horizon_days']} days")
    for day in plan["days"]:
        print(f"Day {day['day']}:")
        for it in day["items"]:
            if it["reps"] is not None:
                print(f"  - {it['workout_name']}: {it['sets']} x {it['reps']} reps"
                      + (f" | notes: {it['notes']}" if it.get('notes') else ""))
            else:
                print(f"  - {it['workout_name']}: {it['sets']} x {it['duration_seconds']}s"
                      + (f" | notes: {it['notes']}" if it.get('notes') else ""))

if __name__ == "__main__":
    main(7)
