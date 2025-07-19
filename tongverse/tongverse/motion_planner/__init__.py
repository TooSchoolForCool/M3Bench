from __future__ import annotations

from tongverse.motion_planner.symbolic_planner import SymbolicPlanner


def get_motion_planner(planner_name: str, env: None, agent: None):
    # "VKC" "Symbolic"
    if planner_name == "symbolic_planner":
        return SymbolicPlanner(env, agent)
    return None
