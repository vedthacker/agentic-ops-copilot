import json
from dataclasses import replace

import pytest

from opscopilot.config import load_settings
from opscopilot.data import build_db
from opscopilot.llm import LLMReply, LLMRequest, RuleBasedLLM
from opscopilot.tools import Toolbox


@pytest.fixture()
def settings(tmp_path):
    s = replace(load_settings(), state_dir=tmp_path / "state", db_path=tmp_path / "ops.db")
    build_db(s.db_path)
    return s


@pytest.fixture()
def toolbox(settings):
    return Toolbox(settings.db_path, settings.runbook_dir, settings.state_dir)


class ScriptedLLM:
    """Replays canned raw replies per role (as strings), so tests can feed malformed JSON, bad tools, etc.
    When a role's script runs out it falls back to the rule-based policy."""

    def __init__(self, script: dict[str, list[str]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.fallback = RuleBasedLLM()
        self.calls: list[str] = []

    def complete(self, req: LLMRequest) -> LLMReply:
        self.calls.append(req.role)
        if self.script.get(req.role):
            text = self.script[req.role].pop(0)
            return LLMReply(text, 100, max(1, len(text) // 4), 1.0)
        return self.fallback.complete(req)


def jd(**kw) -> str:
    return json.dumps(kw)
