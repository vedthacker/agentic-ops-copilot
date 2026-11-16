"""python -m opscopilot.cli run "task" [--approve|--reject|--ask] [--llm rules|gemini]"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from langgraph.types import Command

from .config import load_settings
from .data import build_db
from .graph import open_app
from .runner import auto_approve, auto_reject, run_task


def ask_human(payload: dict) -> dict:
    esc = payload["escalation"]
    print(f"\n[approval needed] Escalate ticket T-{esc['ticket_id']} to {esc['team']}?\n  {esc['summary']}")
    ok = input("approve? [y/N] ").strip().lower() == "y"
    return {"approved": ok, "note": "approved by human" if ok else "rejected by human"}


def show(result: dict) -> None:
    print("\n" + "=" * 70 + f"\nthread={result['thread_id']}  status={result['status']}\n" + "=" * 70)
    print(result.get("draft", ""))
    if result.get("escalation"):
        print("\nproposed escalation:", json.dumps(result["escalation"]))
    for n in result.get("notes", []):
        print("note:", n)
    m = result["metrics"]
    print(f"\n--- run metrics: {m['llm_calls']} LLM calls, {m['tool_calls']} tool calls, {m['revisions']} revision(s), "
          f"{m['input_tokens']}+{m['output_tokens']} tokens (approx), est ${m['est_cost_usd']}, wall {m['wall_ms']} ms")
    for agent, a in m["per_agent"].items():
        print(f"    {agent:<11} llm={a['llm_calls']} tools={a['tool_calls']} tokens={a['tokens']}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="opscopilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--thread", default=None)
    r.add_argument("--llm", choices=["rules", "gemini"], default=None)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    h = sub.add_parser("history", help="list checkpoints of a thread (state-management / time-travel view)")
    h.add_argument("--thread", required=True)
    sub.add_parser("eval")
    args = p.parse_args(argv)
    settings = load_settings()
    if getattr(args, "llm", None):
        settings = replace(settings, llm=args.llm)

    if args.cmd == "seed":
        print("wrote", build_db(settings.db_path))
    elif args.cmd == "run":
        decide = auto_approve if args.approve else auto_reject if args.reject else ask_human
        with open_app(settings) as (graph, _):
            show(run_task(graph, args.task, settings, args.thread, decide))
    elif args.cmd == "history":
        with open_app(settings) as (graph, _):
            cfg = {"configurable": {"thread_id": args.thread}}
            for snap in graph.get_state_history(cfg):
                v = snap.values
                print(f"step={snap.metadata.get('step'):>2} next={list(snap.next)} draft_v={v.get('draft_version', 0)} "
                      f"findings={sorted(v.get('findings', {}))} status={v.get('status')}")
    elif args.cmd == "eval":
        from .eval import main as eval_main

        eval_main(settings)


if __name__ == "__main__":
    main()
