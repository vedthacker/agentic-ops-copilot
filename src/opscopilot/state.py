"""Shared graph state. Reducers decide how parallel branches merge their writes:
  findings  - dict merge (researcher and analyst write different keys at the same time)
  trace     - list concatenation (every node appends events)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class OpsState(TypedDict, total=False):
    task: str
    plan: list[str]                                  # which workers the supervisor decided are needed
    dispatch: list[str]                              # next node(s) to run (can be several => parallel)
    findings: Annotated[dict, merge_dicts]           # worker name -> {"answer": str, "sources": [...]}
    draft: str
    draft_version: int
    critique: list[str]
    critic_ok: bool
    escalation: dict | None                          # proposed side effect, needs human approval
    approval: dict | None
    status: str                                      # running | completed | rejected | budget_exceeded | max_revisions
    notes: Annotated[list, operator.add]             # human-readable guard messages
    trace: Annotated[list, operator.add]
