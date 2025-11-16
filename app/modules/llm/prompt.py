# app/modules/llm/prompt.py
import json
def build_prompt(profile_json: dict, workouts_json: list[dict], horizon_days: int) -> str:
    rules = f"""
You are a workout planning assistant. Generate a {horizon_days}-day plan using ONLY workouts provided.
Rules:
- Use names exactly as listed in available_workouts.
- Strength: sets + reps, duration_seconds = null.
- Time-based: sets + duration_seconds, reps = null.
- Respect user's free_days and equipment.
- Balance load across the week based on goal and fitness_level.
- Keep notes short.
- Output MUST be strict JSON matching the schema. No prose.
""".strip()
    schema = {"WorkoutPlan":
              {
                  "member_id":"int",
                  "title":"string",
                  "days":[{
                      "day":"int>0",
                      "items":[{
                          "workout_name":"string",
                          "sets":"int>0",
                          "reps":"int|null",
                          "duration_seconds":"int|null",
                          "notes":"string|null"
                          }]
                    }]
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
