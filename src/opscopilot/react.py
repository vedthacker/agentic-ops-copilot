"""A ReAct worker as its own LangGraph subgraph:  reason -> act -> reason -> ... -> final.

Each worker (researcher, analyst) is one compiled subgraph with its own tool list and step limit.
The parent graph treats it as a black box, which is the "hierarchical delegation" part: the supervisor
never sees tool calls, only the worker's final answer.

Guardrails inside the loop:
  * unknown tool name / bad arguments  -> ERROR observation, the model gets another turn to fix it
  * invalid JSON from the model        -> one corrective retry, then the worker gives up cleanly
  * step limit                         -> the worker is forced to answer from what it has gathered
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import LLM, LLMRequest, parse_json
from .tools import Toolbox
from .trace import event

SYSTEM = """You are the {role} agent in a support-operations team.
Goal: {goal_hint}
Use the ReAct pattern. Reply with ONE JSON object per turn:
  {{"thought": "...", "action": "<tool name>", "action_input": {{...}}}}   to call a tool, or
  {{"thought": "...", "action": "final", "answer": "...", "sources": ["runbook or table names"]}}  when done.
Available tools:
{tools}
Rules: only use the tools listed; never invent ticket ids, customers or numbers - if a lookup finds nothing, say so."""

HINTS = {
    "researcher": "find the policy / runbook guidance that applies to the task.",
    "analyst": "look up the customer and ticket data that applies to the task.",
}


class ReactState(TypedDict, total=False):
    task: str
    scratch: list[dict]
    pending: dict | None
    answer: str
    sources: list
    steps: int
    done: bool
    trace: Annotated[list, operator.add]   # each node appends its events


def _prompt(task: str, scratch: list[dict]) -> str:
    history = "\n".join(
        f"Thought: {s['thought']}\nAction: {s['action']} {s['input']}\nObservation: {s['observation']}" for s in scratch)
    return f"Task: {task}\n\n{history}\n\nNext JSON action:"


def build_worker(role: str, tool_names: list[str], llm: LLM, toolbox: Toolbox, settings: Settings):
    system = SYSTEM.format(role=role, goal_hint=HINTS[role], tools=toolbox.specs(tool_names))

    def reason(state: ReactState) -> dict:
        scratch, steps = state.get("scratch", []), state.get("steps", 0)
        trace: list[dict] = []
        if steps >= settings.max_react_steps:
            trace.append(event(role, "guard", "max_steps", detail=f"forced final after {steps} steps"))
            return {"done": True, "pending": None, "trace": trace,
                    "answer": "\n".join(s["observation"] for s in scratch) or "No data gathered.",
                    "sources": []}
        prompt = _prompt(state["task"], scratch)
        for attempt in range(2):
            reply = llm.complete(LLMRequest(role, system, prompt, {"task": state["task"], "scratch": scratch}))
            trace.append(event(role, "llm", "reason", reply.ms, reply.in_tokens, reply.out_tokens))
            try:
                out = parse_json(reply.text)
                break
            except ValueError as exc:
                trace.append(event(role, "guard", "invalid_json", detail=str(exc)))
                prompt += "\n\nYour previous reply was not valid JSON. Reply with exactly one JSON object."
        else:
            return {"done": True, "pending": None, "trace": trace, "answer": "Agent could not produce a valid action.",
                    "sources": []}
        if out.get("action") == "final":
            return {"done": True, "pending": None, "trace": trace, "answer": str(out.get("answer", "")),
                    "sources": list(out.get("sources", []))}
        return {"done": False, "pending": {"thought": out.get("thought", ""), "action": out.get("action", ""),
                                           "input": out.get("action_input", {})}, "trace": trace}

    def act(state: ReactState) -> dict:
        p = state["pending"]
        t = time.perf_counter()
        if p["action"] not in tool_names:
            obs = f"ERROR: unknown tool '{p['action']}'. Available: {tool_names}"
        else:
            obs = toolbox.tools[p["action"]].call(p["input"])
        ms = (time.perf_counter() - t) * 1000
        return {"scratch": [*state.get("scratch", []), {**p, "observation": obs}], "steps": state.get("steps", 0) + 1,
                "pending": None, "trace": [event(role, "tool", p["action"], ms, detail=obs[:200])]}

    g = StateGraph(ReactState)
    g.add_node("reason", reason)
    g.add_node("act", act)
    g.add_edge(START, "reason")
    g.add_conditional_edges("reason", lambda s: END if s.get("done") else "act", {END: END, "act": "act"})
    g.add_edge("act", "reason")
    return g.compile()


def worker_node(role: str, tool_names: list[str], llm: LLM, toolbox: Toolbox, settings: Settings):
    """Wrap the subgraph as a node of the parent graph: state in -> findings + trace out."""
    sub = build_worker(role, tool_names, llm, toolbox, settings)

    def run(state: dict) -> dict:
        result = sub.invoke({"task": state["task"], "scratch": [], "steps": 0, "trace": []},
                            {"recursion_limit": settings.max_react_steps * 2 + 6})
        return {"findings": {role: {"answer": result.get("answer", ""), "sources": result.get("sources", []),
                                    "steps": result.get("steps", 0)}},
                "trace": result.get("trace", [])}

    return run
