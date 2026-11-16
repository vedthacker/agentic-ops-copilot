r"""Assemble the parent graph.

        START
          |
      supervisor  <-----------+
       /       \               |
  researcher  analyst  (parallel fan-out when both are planned)
       \       /
        supervisor  --(all reported)--> drafter <------+
                                           |            |
                                         critic --(issues, revisions left)
                                           |
                                      human_gate  (interrupt before any side effect)
                                           |
                                          END
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .agents import (make_critic, make_drafter, make_human_gate, make_route_critic, make_supervisor,
                     route_supervisor)
from .config import Settings, load_settings
from .data import build_db
from .llm import LLM, GeminiLLM, RuleBasedLLM
from .react import worker_node
from .state import OpsState
from .tools import Toolbox


def make_llm(settings: Settings) -> LLM:
    return GeminiLLM(settings) if settings.llm == "gemini" else RuleBasedLLM()


def build_graph(settings: Settings, llm: LLM, toolbox: Toolbox, checkpointer=None):
    g = StateGraph(OpsState)
    g.add_node("supervisor", make_supervisor(llm, settings))
    g.add_node("researcher", worker_node("researcher", ["search_runbooks", "get_sla"], llm, toolbox, settings))
    g.add_node("analyst", worker_node("analyst", ["get_customer", "get_sla", "list_tickets", "ticket_stats"], llm, toolbox, settings))
    g.add_node("drafter", make_drafter(llm, settings))
    g.add_node("critic", make_critic(llm, settings))
    g.add_node("human_gate", make_human_gate(toolbox))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_supervisor, ["researcher", "analyst", "drafter"])
    g.add_edge("researcher", "supervisor")
    g.add_edge("analyst", "supervisor")
    g.add_edge("drafter", "critic")
    g.add_conditional_edges("critic", make_route_critic(settings), ["drafter", "human_gate"])
    g.add_edge("human_gate", END)
    return g.compile(checkpointer=checkpointer)


@contextmanager
def open_app(settings: Settings | None = None, llm: LLM | None = None):
    """Yield a compiled graph wired to a SQLite checkpointer, creating the demo DB on first use."""
    settings = settings or load_settings()
    if not settings.db_path.exists():
        build_db(settings.db_path)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    toolbox = Toolbox(settings.db_path, settings.runbook_dir, settings.state_dir)
    with SqliteSaver.from_conn_string(str(Path(settings.state_dir) / "checkpoints.sqlite")) as saver:
        yield build_graph(settings, llm or make_llm(settings), toolbox, saver), toolbox
