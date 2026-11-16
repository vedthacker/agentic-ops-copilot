"""Granular tracing. Every LLM call, tool call and routing decision becomes one event in the graph
state, so a checkpointed run carries its own trace and can be inspected or resumed later.

Event fields: ts, agent, kind (llm | tool | route | guard), name, ms, in_tokens, out_tokens, detail.
"""

from __future__ import annotations

import time
from collections import defaultdict


def event(agent: str, kind: str, name: str, ms: float = 0.0, in_tokens: int = 0, out_tokens: int = 0,
          detail: str = "") -> dict:
    return {"ts": round(time.time(), 3), "agent": agent, "kind": kind, "name": name, "ms": round(ms, 2),
            "in_tokens": in_tokens, "out_tokens": out_tokens, "detail": detail[:300]}


def tokens_used(trace: list[dict]) -> int:
    return sum(e["in_tokens"] + e["out_tokens"] for e in trace)


def summarize(trace: list[dict], price_in_per_m: float, price_out_per_m: float) -> dict:
    """LLM-native metrics for one run: calls, tokens, estimated cost, per-agent breakdown."""
    llm = [e for e in trace if e["kind"] == "llm"]
    tools = [e for e in trace if e["kind"] == "tool"]
    tin, tout = sum(e["in_tokens"] for e in llm), sum(e["out_tokens"] for e in llm)
    per_agent: dict[str, dict] = defaultdict(lambda: {"llm_calls": 0, "tool_calls": 0, "tokens": 0, "ms": 0.0})
    for e in trace:
        a = per_agent[e["agent"]]
        a["llm_calls"] += e["kind"] == "llm"
        a["tool_calls"] += e["kind"] == "tool"
        a["tokens"] += e["in_tokens"] + e["out_tokens"]
        a["ms"] = round(a["ms"] + e["ms"], 1)
    out_ms = sum(e["ms"] for e in llm)
    return {
        "llm_calls": len(llm), "tool_calls": len(tools), "input_tokens": tin, "output_tokens": tout,
        "est_cost_usd": round(tin / 1e6 * price_in_per_m + tout / 1e6 * price_out_per_m, 6),
        "output_tokens_per_sec": round(tout / (out_ms / 1000), 1) if out_ms else None,
        "tool_errors": sum(1 for e in tools if e["detail"].startswith("ERROR")),
        "per_agent": dict(per_agent),
    }
