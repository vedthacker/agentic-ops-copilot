from dataclasses import replace

from conftest import ScriptedLLM, jd
from opscopilot.agents import grounding_issues, rubric_issues
from opscopilot.graph import open_app
from opscopilot.llm import RuleBasedLLM
from opscopilot.react import build_worker


def run_worker(settings, toolbox, llm, role="researcher", tools=("search_runbooks", "get_sla")):
    w = build_worker(role, list(tools), llm, toolbox, settings)
    return w.invoke({"task": "HTTP 429 errors", "scratch": [], "steps": 0, "trace": []})


def test_invalid_json_gets_one_corrective_retry(settings, toolbox):
    llm = ScriptedLLM({"researcher": ["I think we should search", jd(thought="ok", action="final", answer="done", sources=[])]})
    out = run_worker(settings, toolbox, llm)
    assert out["answer"] == "done"
    assert any(e["name"] == "invalid_json" for e in out["trace"])


def test_two_invalid_replies_end_the_worker_cleanly(settings, toolbox):
    out = run_worker(settings, toolbox, ScriptedLLM({"researcher": ["nope", "still nope"]}))
    assert "could not produce a valid action" in out["answer"]


def test_agent_recovers_from_unknown_tool_and_bad_arguments(settings, toolbox):
    llm = ScriptedLLM({"researcher": [
        jd(thought="t", action="delete_everything", action_input={}),
        jd(thought="t", action="get_sla", action_input={"tier": 3}),
        jd(thought="t", action="get_sla", action_input={"tier": "Business"}),
        jd(thought="t", action="final", answer="Business SLA is 1 hour", sources=[])]})
    out = run_worker(settings, toolbox, llm)
    obs = [s["observation"] for s in out["scratch"]]
    assert obs[0].startswith("ERROR: unknown tool") and obs[1].startswith("ERROR: argument 'tier'")
    assert "first response" in obs[2] and out["answer"].startswith("Business SLA")
    assert sum(e["detail"].startswith("ERROR") for e in out["trace"] if e["kind"] == "tool") == 2


def test_step_limit_forces_an_answer(settings, toolbox):
    settings = replace(settings, max_react_steps=3)
    loop = jd(thought="again", action="get_sla", action_input={"tier": "Basic"})
    out = run_worker(settings, toolbox, ScriptedLLM({"researcher": [loop] * 10}))
    assert out["steps"] == 3 and any(e["name"] == "max_steps" for e in out["trace"])
    assert "first response" in out["answer"]            # forced final still carries what was gathered


def test_grounding_guard_flags_invented_ticket_ids():
    findings = {"analyst": {"answer": "T-8 SEV3 open [api] Echo Motors"}}
    assert grounding_issues("See T-8 for details", findings) == []
    assert "T-999" in grounding_issues("See T-999", findings)[0]


def test_rubric_flags_missing_sections_and_fix_time_promises():
    issues = rubric_issues("## Summary\nIt will be fixed by tomorrow.")
    assert any("Next steps" in i for i in issues) and any("fix time" in i for i in issues)


def test_token_budget_stops_delegation_but_still_produces_a_draft(settings):
    from opscopilot.runner import run_task

    s = replace(settings, token_budget=300)
    with open_app(s) as (graph, _):
        r = run_task(graph, "Customer Echo Motors reports HTTP 429 errors.", s)
    assert r["status"] == "budget_exceeded" and r["draft"] and any("budget" in n.lower() for n in r["notes"])


def test_revision_limit_is_enforced(settings):
    class Stubborn(RuleBasedLLM):
        def _drafter(self, v):
            return {"draft": "## Summary\nonly a summary", "escalation": None}

    from opscopilot.runner import run_task

    with open_app(settings, Stubborn()) as (graph, _):
        r = run_task(graph, "What approvals are needed for a 1,200 USD refund?", settings)
    assert r["status"] == "max_revisions" and r["draft_version"] == settings.max_revisions + 1
