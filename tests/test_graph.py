import json

from langgraph.types import Command

from opscopilot.graph import open_app
from opscopilot.runner import auto_approve, auto_reject, run_task

OUTAGE = "Customer Cedar Foods says the tracking dashboard is down in EMEA. What is our procedure?"


def test_supervisor_fans_out_to_both_workers_in_parallel(settings):
    with open_app(settings) as (graph, _):
        r = run_task(graph, "Customer Echo Motors reports HTTP 429 errors on /v1/rates.", settings)
    assert set(r["findings"]) == {"researcher", "analyst"}
    sup = [e for e in r["trace"] if e["agent"] == "supervisor" and e["kind"] == "route"]
    assert sup[0]["name"] == "researcher,analyst" and sup[-1]["name"] == "drafter"
    assert r["metrics"]["revisions"] >= 1               # the first draft really was revised by the critic


def test_policy_question_skips_the_data_analyst(settings):
    with open_app(settings) as (graph, _):
        r = run_task(graph, "What approvals are needed for a 1,200 USD refund?", settings)
    assert set(r["findings"]) == {"researcher"} and "manager approval" in r["draft"]


def test_side_effect_waits_for_human_and_runs_only_when_approved(settings):
    with open_app(settings) as (graph, tb):
        r = run_task(graph, OUTAGE, settings, decide=auto_approve)
    esc = (settings.state_dir / "escalations.jsonl").read_text().splitlines()
    assert r["status"] == "completed" and json.loads(esc[0])["ticket_id"] == 27


def test_rejected_escalation_has_no_side_effect(settings):
    with open_app(settings) as (graph, _):
        r = run_task(graph, OUTAGE, settings, decide=auto_reject)
    assert r["status"] == "rejected" and not (settings.state_dir / "escalations.jsonl").exists()


def test_paused_run_survives_a_process_restart(settings):
    """Interrupt, throw the whole graph away, reopen the SQLite checkpoint, resume - same thread, same state."""
    cfg = {"configurable": {"thread_id": "durable"}, "recursion_limit": 40}
    with open_app(settings) as (graph, _):
        paused = graph.invoke({"task": OUTAGE, "status": "running"}, cfg)
        assert paused["__interrupt__"][0].value["escalation"]["ticket_id"] == 27
        assert not (settings.state_dir / "escalations.jsonl").exists()          # nothing executed yet
    with open_app(settings) as (graph, _):                                      # "new process"
        assert graph.get_state(cfg).next == ("human_gate",)
        done = graph.invoke(Command(resume={"approved": True, "note": "ok"}), cfg)
    assert done["status"] == "completed" and (settings.state_dir / "escalations.jsonl").exists()


def test_checkpoint_history_is_queryable(settings):
    with open_app(settings) as (graph, _):
        run_task(graph, "What approvals are needed for a 1,200 USD refund?", settings, thread_id="hist")
        history = list(graph.get_state_history({"configurable": {"thread_id": "hist"}}))
    assert len(history) >= 6 and history[0].values["status"] == "completed"


def test_every_run_writes_a_jsonl_trace(settings):
    with open_app(settings) as (graph, _):
        r = run_task(graph, "What approvals are needed for a 1,200 USD refund?", settings, thread_id="tr")
    lines = (settings.state_dir / "traces" / "tr.jsonl").read_text().splitlines()
    assert len(lines) == len(r["trace"]) and {"agent", "kind", "in_tokens"} <= set(json.loads(lines[0]))


def test_scenario_suite_passes_offline(settings):
    from opscopilot.eval import evaluate

    rep = evaluate(settings)
    failed = [(s["scenario"], k) for s in rep["scenarios"] for k, ok in s["checks"].items() if not ok]
    assert not failed, failed
