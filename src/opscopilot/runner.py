"""Run a task end to end, including the pause for human approval."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

from langgraph.types import Command

from .config import Settings
from .trace import summarize

Decide = Callable[[dict], dict]


def auto_approve(_payload: dict) -> dict:
    return {"approved": True, "note": "auto-approved"}


def auto_reject(_payload: dict) -> dict:
    return {"approved": False, "note": "auto-rejected"}


def run_task(graph, task: str, settings: Settings, thread_id: str | None = None, decide: Decide = auto_approve) -> dict:
    thread_id = thread_id or uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    result = graph.invoke({"task": task, "status": "running"}, cfg)
    paused = 0
    while result.get("__interrupt__"):          # human-in-the-loop: the graph is checkpointed and waiting
        paused += 1
        result = graph.invoke(Command(resume=decide(result["__interrupt__"][0].value)), cfg)
        if paused > 3:
            break
    result["thread_id"] = thread_id
    result["metrics"] = summarize(result.get("trace", []), settings.price_in_per_m, settings.price_out_per_m)
    result["metrics"]["revisions"] = max(0, result.get("draft_version", 1) - 1)
    result["metrics"]["wall_ms"] = (round((result["trace"][-1]["ts"] - result["trace"][0]["ts"]) * 1000, 1)
                                     if result.get("trace") else 0)
    save_trace(settings.state_dir, thread_id, result.get("trace", []))
    return result


def save_trace(state_dir: Path, thread_id: str, trace: list[dict]) -> Path:
    path = Path(state_dir) / "traces" / f"{thread_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in trace) + "\n")
    return path
