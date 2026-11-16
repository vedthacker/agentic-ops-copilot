"""Scenario evaluation: does the whole system behave correctly on realistic tasks?

IMPORTANT: with the default offline RuleBasedLLM this measures the *plumbing* (routing, parallel
delegation, tool use, reflection loop, guardrails, human approval, state), not language-model quality.
Run with `--llm gemini` to evaluate a real model against the same checks.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .config import Settings, load_settings
from .graph import open_app
from .runner import auto_approve, auto_reject, run_task


@dataclass
class Scenario:
    name: str
    task: str
    decide: Callable = auto_approve
    checks: dict[str, Callable[[dict, "Ctx"], bool]] = field(default_factory=dict)


@dataclass
class Ctx:
    settings: Settings

    def sql(self, q: str):
        con = sqlite3.connect(f"file:{self.settings.db_path}?mode=ro", uri=True)
        try:
            return con.execute(q).fetchall()
        finally:
            con.close()

    def escalations(self) -> list[dict]:
        f = Path(self.settings.state_dir) / "escalations.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def _has(result, *words):          # all words present in the draft (case-insensitive)
    d = result.get("draft", "").lower()
    return all(w.lower() in d for w in words)


def _workers(result):
    return set(result.get("findings", {}))


SCENARIOS = [
    Scenario("api_429_basic_customer",
             "Customer Echo Motors reports HTTP 429 errors on /v1/rates. Triage and draft a response.",
             checks={"both workers used": lambda r, c: _workers(r) == {"researcher", "analyst"},
                     "cites the customer's real ticket": lambda r, c: _has(r, "T-8"),
                     "gives backoff advice": lambda r, c: _has(r, "backoff"),
                     "uses the Basic SLA": lambda r, c: _has(r, "Basic", "8 business hours"),
                     "all required sections": lambda r, c: not __import__("opscopilot.agents", fromlist=["x"]).rubric_issues(r["draft"]),
                     "no escalation for a SEV3": lambda r, c: not r.get("escalation") and not c.escalations()}),
    Scenario("sev1_outage_approved",
             "Customer Cedar Foods says the tracking dashboard is down in EMEA. What is our procedure and who needs to be involved?",
             checks={"finds the SEV1 ticket": lambda r, c: _has(r, "T-27", "SEV1"),
                     "quotes 5 minute acknowledgement": lambda r, c: _has(r, "5 minutes"),
                     "proposes Platform on-call escalation": lambda r, c: (r.get("escalation") or {}).get("team") == "Platform on-call",
                     "escalation executed after approval": lambda r, c: any(e["ticket_id"] == 27 for e in c.escalations()),
                     "status completed": lambda r, c: r["status"] == "completed"}),
    Scenario("sev1_outage_rejected",
             "Customer Cedar Foods says the tracking dashboard is down in EMEA. What is our procedure and who needs to be involved?",
             decide=auto_reject,
             checks={"status rejected": lambda r, c: r["status"] == "rejected",
                     "side effect NOT executed": lambda r, c: not c.escalations()}),
    Scenario("open_ticket_stats",
             "How many open tickets do we have, broken down by severity?",
             checks={"counts match the database": lambda r, c: all(
                 f"{sev}={n}" in r["draft"] for sev, n in c.sql(
                     "SELECT severity, COUNT(*) FROM tickets WHERE status='open' GROUP BY severity ORDER BY 2 DESC"))}),
    Scenario("policy_only_skips_analyst",
             "What approvals are needed for a 1,200 USD refund?",
             checks={"only the researcher ran": lambda r, c: _workers(r) == {"researcher"},
                     "answers from policy": lambda r, c: _has(r, "manager approval")}),
    Scenario("unknown_customer_no_hallucination",
             "Customer Zeta Corp reports slow dashboards. Triage this.",
             checks={"says customer not found": lambda r, c: _has(r, "was not found"),
                     "invents no ticket ids": lambda r, c: "T-" not in r["draft"],
                     "no escalation": lambda r, c: not r.get("escalation")}),
]


def evaluate(settings: Settings, scenarios=SCENARIOS) -> dict:
    rows, totals = [], {"passed": 0, "checks": 0}
    for sc in scenarios:
        with tempfile.TemporaryDirectory() as tmp:        # isolated state per scenario (checkpoints, escalations)
            s = replace(settings, state_dir=Path(tmp), db_path=settings.db_path)
            with open_app(s) as (graph, _):
                result = run_task(graph, sc.task, s, decide=sc.decide)
            ctx = Ctx(s)
            outcome = {name: bool(fn(result, ctx)) for name, fn in sc.checks.items()}
        m = result["metrics"]
        rows.append({"scenario": sc.name, "checks": outcome, "llm_calls": m["llm_calls"], "tool_calls": m["tool_calls"],
                     "revisions": m["revisions"], "tokens": m["input_tokens"] + m["output_tokens"],
                     "est_cost_usd": m["est_cost_usd"], "status": result["status"]})
        totals["passed"] += sum(outcome.values())
        totals["checks"] += len(outcome)
    return {"llm": settings.llm, "scenarios": rows, **totals}


def main(settings: Settings | None = None) -> None:
    rep = evaluate(settings or load_settings())
    print(f"llm backend: {rep['llm']}\n")
    for r in rep["scenarios"]:
        ok = sum(r["checks"].values())
        print(f"{r['scenario']:<36} {ok}/{len(r['checks'])} checks  llm={r['llm_calls']:>2} tools={r['tool_calls']} "
              f"rev={r['revisions']} tokens={r['tokens']} status={r['status']}")
        for name, passed in r["checks"].items():
            if not passed:
                print(f"    FAILED: {name}")
    print(f"\nTOTAL {rep['passed']}/{rep['checks']} checks passed")


if __name__ == "__main__":
    main()
