from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph
from app.modules.llm.prompt import build_prompt
from app.modules.llm.cohere_chain import run_llm
from app.modules.llm.validate import validate_plan

class State(TypedDict):
    profile: Dict
    workouts: List[Dict]
    horizon_days: int
    llm_output: Dict
    plan_validated: Dict

def node_build_prompt(state: State) -> State:
    prompt = build_prompt(state["profile"], state["workouts"], state["horizon_days"])
    state["llm_output"] = run_llm(prompt); return state

def node_validate(state: State) -> State:
    allowed = {w["name"] for w in state["workouts"]}
    validated = validate_plan(state["llm_output"], allowed)
    state["plan_validated"] = validated.model_dump(); return state

def build_graph():
    g = StateGraph(State)
    g.add_node("build_prompt", node_build_prompt)
    g.add_node("validate", node_validate)
    g.add_edge("build_prompt", "validate")
    g.set_entry_point("build_prompt"); g.set_finish_point("validate")
    return g.compile()
