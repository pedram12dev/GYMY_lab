# lab/llm/prompt.py
import json

def build_prompt(profile_json: dict, workouts_json: list[dict], horizon_days: int) -> str:
    rules = """
You are a workout planning assistant. Generate a {horizon}-day plan using ONLY workouts provided.
Rules:
- Use names exactly as listed in available_workouts.
- Strength: sets + reps, duration_seconds = null.
- Time-based: sets + duration_seconds, reps = null.
- Respect user's free_days and equipment.
- Balance load across the week based on goal and fitness_level.
- Keep notes short.
- Output MUST be strict JSON matching the schema. No prose.
""".strip().format(horizon=horizon_days)

    schema = {
        "type": "object",
        "required": ["profile_id", "horizon_days", "days"],
        "properties": {
            "profile_id": {"type": "integer"},
            "horizon_days": {"type": "integer"},
            "days": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["day", "items"],
                    "properties": {
                        "day": {"type": "integer"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["workout_name", "sets"],
                                "properties": {
                                    "workout_name": {"type": "string"},
                                    "sets": {"type": "integer"},
                                    "reps": {"type": ["integer", "null"]},
                                    "duration_seconds": {"type": ["integer", "null"]},
                                    "notes": {"type": ["string", "null"]}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return f"""
{rules}

Schema:
{json.dumps(schema, indent=2)}

Profile:
{json.dumps(profile_json, indent=2)}

Available workouts:
{json.dumps(workouts_json, indent=2)}

Return ONLY the JSON.
""".strip()
