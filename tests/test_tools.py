import json

import pytest

from opscopilot.llm import parse_json


def test_validation_errors_become_observations_not_exceptions(toolbox):
    t = toolbox.tools
    assert t["get_sla"].call({}).startswith("ERROR: missing required argument 'tier'")
    assert t["get_sla"].call({"tier": 5}).startswith("ERROR: argument 'tier' must be str")
    assert t["get_sla"].call({"tier": "Gold"}).startswith("ERROR: unknown tier")
    assert "unknown argument" in t["get_customer"].call({"name": "x", "evil": 1})
    assert t["list_tickets"].call({"limit": True}).startswith("ERROR")          # bool is not an int here
    assert t["list_tickets"].call({"severity": "SEV9"}).startswith("ERROR: severity must be one of")


def test_sql_injection_through_tool_arguments_is_inert(toolbox):
    evil = "x'; DROP TABLE tickets; --"
    assert toolbox.tools["get_customer"].call({"name": evil}).startswith("NOT_FOUND")
    assert toolbox.tools["ticket_stats"].call({"group_by": "severity; DROP TABLE tickets"}).startswith("ERROR")
    assert toolbox.tools["list_tickets"].call({"customer": evil}) == "NO_TICKETS"
    assert toolbox._q("SELECT COUNT(*) FROM tickets")[0][0] == 90                  # still there


def test_database_connection_is_read_only(toolbox):
    with pytest.raises(Exception):
        toolbox._q("DELETE FROM tickets")


def test_side_effect_tool_is_not_exposed_to_workers(toolbox):
    assert "create_escalation" not in toolbox.tools
    toolbox.create_escalation(1, "Support Lead", "x")
    assert json.loads((toolbox.state_dir / "escalations.jsonl").read_text())["ticket_id"] == 1


def test_runbook_search_ranks_the_right_section(toolbox):
    out = toolbox.tools["search_runbooks"].call({"query": "HTTP 429 backoff"})
    assert out.splitlines()[0].startswith("[api_integration_runbook]")
    assert toolbox.tools["search_runbooks"].call({"query": "zzzqqq"}) == "NO_RESULTS"


@pytest.mark.parametrize("raw", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go: {"a": 1} hope it helps'])
def test_parse_json_tolerates_fences_and_prose(raw):
    assert parse_json(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_parse_json_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_json(raw)
