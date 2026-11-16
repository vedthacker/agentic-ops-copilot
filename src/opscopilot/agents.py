"""Parent-graph nodes: supervisor, drafter, critic (self-reflection) and the human approval gate."""

from __future__ import annotations

import re
import time

from langgraph.types import interrupt

from .config import Settings
from .llm import LLM, LLMRequest, parse_json
from .tools import Toolbox
from .trace import event, tokens_used

REQUIRED_SECTIONS = ["Summary", "Severity and SLA", "Next steps", "Likely root cause", "Customer message", "Sources"]
WORKERS = {"researcher", "analyst"}
MAX_SUPERVISOR_VISITS = 4

SUPERVISOR_SYSTEM = """You are the supervisor of a support-operations agent team.
Workers: researcher (runbooks and policies), analyst (customer and ticket data), drafter (writes the response).
Reply with ONE JSON object: {"thought": "...", "plan": ["researcher", "analyst"], "dispatch": ["..."]}.
First turn: choose the plan (workers needed, one or both) and dispatch them. Later turns: dispatch any plan
worker that has not reported; once all have reported, dispatch ["drafter"]. Never dispatch a worker twice."""

DRAFTER_SYSTEM = """You write the final support response from the workers' findings ONLY. Never invent ticket ids,
customers, numbers or policies. Reply with ONE JSON object: {"draft": "<markdown>", "escalation": null or
{"ticket_id": int, "team": "...", "summary": "..."}}.
The markdown must contain these sections in order: """ + ", ".join(f"## {s}" for s in REQUIRED_SECTIONS) + """.
Propose an escalation only for an open SEV1/SEV2 or security ticket that appears in the findings."""

CRITIC_SYSTEM = """You review a support response draft. Reply with ONE JSON object: {"ok": true|false, "issues": ["..."]}.
Check: every required section is present (""" + ", ".join(REQUIRED_SECTIONS) + """); next steps are concrete;
the draft promises no fix time; nothing in the draft contradicts the findings."""


def rubric_issues(draft: str) -> list[str]:
    """Deterministic structural rubric - runs in code whatever model is used."""
    issues = []
    for sec in REQUIRED_SECTIONS:
        if not re.search(rf"^##\s+{re.escape(sec)}\b", draft, re.M | re.I):
            issues.append(f"missing section: {sec}" + (" (root cause)" if sec == "Likely root cause" else ""))
    if re.search(r"\b(will be fixed (by|within)|fixed in \d+|eta of \d+)\b", draft, re.I):
        issues.append("draft promises a fix time that engineering has not confirmed")
    return issues


def grounding_issues(draft: str, findings: dict) -> list[str]:
    """Every ticket id in the draft must appear in a worker's findings - blocks invented tickets."""
    evidence = " ".join(w.get("answer", "") for w in findings.values())
    return [f"ticket T-{t} is not in the findings (possible hallucination)"
            for t in sorted(set(re.findall(r"\bT-(\d+)\b", draft))) if f"T-{t} " not in evidence + " "]


def _call(llm: LLM, agent: str, system: str, prompt: str, view: dict, trace: list[dict]) -> dict | None:
    """One JSON-returning LLM call with a single corrective retry. Returns None if the model never produced JSON."""
    for _ in range(2):
        reply = llm.complete(LLMRequest(agent, system, prompt, view))
        trace.append(event(agent, "llm", "complete", reply.ms, reply.in_tokens, reply.out_tokens))
        try:
            return parse_json(reply.text)
        except ValueError as exc:
            trace.append(event(agent, "guard", "invalid_json", detail=str(exc)))
            prompt += "\n\nYour previous reply was not valid JSON. Reply with exactly one JSON object."
    return None


def make_supervisor(llm: LLM, settings: Settings):
    def supervisor(state: dict) -> dict:
        trace: list[dict] = []
        visits = sum(1 for e in state.get("trace", []) if e["agent"] == "supervisor" and e["kind"] == "route")
        findings, plan = state.get("findings", {}), state.get("plan", [])

        if tokens_used(state.get("trace", [])) > settings.token_budget and "drafter" not in state.get("dispatch", []):
            trace.append(event("supervisor", "guard", "token_budget", detail=f"budget {settings.token_budget} exceeded"))
            return {"dispatch": ["drafter"], "status": "budget_exceeded", "trace": trace,
                    "notes": ["Token budget exceeded: drafting from partial findings."]}
        if visits >= MAX_SUPERVISOR_VISITS:
            trace.append(event("supervisor", "guard", "max_visits"))
            return {"dispatch": ["drafter"], "trace": trace, "notes": ["Supervisor loop limit reached."]}

        prompt = (f"Task: {state['task']}\nPlan so far: {plan}\nWorkers that have reported: {sorted(findings)}\n"
                  "Next JSON decision:")
        out = _call(llm, "supervisor", SUPERVISOR_SYSTEM, prompt,
                    {"task": state["task"], "plan": plan, "findings": findings}, trace) or {}
        dispatch = [d for d in out.get("dispatch", []) if d in WORKERS | {"drafter"}]
        new_plan = [p for p in out.get("plan", plan) if p in WORKERS] or plan
        if not dispatch:   # model failed or gave nonsense: fall back to a safe default
            dispatch = [w for w in (new_plan or ["researcher"]) if w not in findings] or ["drafter"]
            trace.append(event("supervisor", "guard", "fallback_dispatch", detail=str(dispatch)))
        dispatch = [d for d in dispatch if d == "drafter" or d not in findings]   # never re-run a finished worker
        if not dispatch:
            dispatch = ["drafter"]
        trace.append(event("supervisor", "route", ",".join(dispatch), detail=out.get("thought", "")))
        return {"plan": new_plan or ["researcher"], "dispatch": dispatch, "trace": trace}

    return supervisor


def route_supervisor(state: dict):
    """Returns a LIST when several workers are dispatched: LangGraph runs them in parallel."""
    d = state["dispatch"]
    return d if len(d) > 1 else d[0]


def make_drafter(llm: LLM, settings: Settings):
    def drafter(state: dict) -> dict:
        trace: list[dict] = []
        version = state.get("draft_version", 0) + 1
        crit = state.get("critique") or []
        findings = state.get("findings", {})
        prompt = (f"Task: {state['task']}\n\nFindings:\n" +
                  "\n".join(f"- {k}: {v.get('answer', '')}\n  sources: {v.get('sources', [])}" for k, v in findings.items()))
        if crit:
            prompt += "\n\nYour previous draft:\n" + state.get("draft", "") + "\n\nFix these issues:\n- " + "\n- ".join(crit)
        out = _call(llm, "drafter", DRAFTER_SYSTEM, prompt,
                    {"task": state["task"], "findings": findings, "critique": crit, "draft": state.get("draft", "")}, trace)
        if not out or not out.get("draft"):
            trace.append(event("drafter", "guard", "no_draft"))
            return {"draft": state.get("draft", "(no draft produced)"), "draft_version": version, "trace": trace,
                    "notes": ["Drafter failed to return a draft."]}
        return {"draft": out["draft"], "draft_version": version, "escalation": out.get("escalation") or state.get("escalation"),
                "trace": trace}

    return drafter


def make_critic(llm: LLM, settings: Settings):
    def critic(state: dict) -> dict:
        trace: list[dict] = []
        draft, findings = state.get("draft", ""), state.get("findings", {})
        out = _call(llm, "critic", CRITIC_SYSTEM, f"Draft:\n{draft}", {"draft": draft, "findings": findings}, trace) or {}
        issues = list(dict.fromkeys(                        # LLM review + hard-coded rubric + grounding guard
            [str(i) for i in out.get("issues", [])] + rubric_issues(draft) + grounding_issues(draft, findings)))
        trace.append(event("critic", "route", "revise" if issues else "approve", detail="; ".join(issues)))
        return {"critique": issues, "critic_ok": not issues, "trace": trace}

    return critic


def make_route_critic(settings: Settings):
    def route_critic(state: dict) -> str:
        if state.get("critic_ok"):
            return "human_gate"
        if state.get("draft_version", 1) > settings.max_revisions:   # revisions exhausted
            return "human_gate"
        return "drafter"

    return route_critic


def make_human_gate(toolbox: Toolbox):
    def human_gate(state: dict) -> dict:
        status = state.get("status", "running")
        notes: list[str] = []
        if not state.get("critic_ok"):
            status = status if status == "budget_exceeded" else "max_revisions"
            notes.append(f"Draft still had open issues after {state.get('draft_version', 0)} version(s): {state.get('critique')}")
        esc = state.get("escalation")
        if not esc:
            return {"status": "completed" if status == "running" else status, "notes": notes,
                    "trace": [event("human_gate", "route", "no_escalation")]}
        # Side-effect-free up to here, so re-running the node after resume is safe.
        decision = interrupt({"type": "approve_escalation", "escalation": esc, "draft": state.get("draft", "")})
        t = time.perf_counter()
        if decision.get("approved"):
            msg = toolbox.create_escalation(esc["ticket_id"], esc["team"], esc["summary"])
            return {"approval": decision, "status": "completed" if status == "running" else status, "notes": notes + [msg],
                    "trace": [event("human_gate", "tool", "create_escalation", (time.perf_counter() - t) * 1000, detail=msg)]}
        return {"approval": decision, "status": "rejected", "notes": notes + [f"Escalation rejected: {decision.get('note', '')}"],
                "trace": [event("human_gate", "route", "rejected", detail=decision.get("note", ""))]}

    return human_gate
