"""Tools the agents can call.

Design rules I settled on:
  * Tools take typed arguments and are validated BEFORE running. A bad call returns an ERROR string
    as the observation - that is how an agent self-corrects - instead of raising and killing the run.
  * Nothing here builds SQL from model text. The model picks a tool and parameters; SQL stays parameterised
    and column names come from allow-lists.
  * `create_escalation` is a side effect, so it is NOT in the worker tool lists. Only the human-approval
    gate can call it.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SEVERITIES = {"SEV1", "SEV2", "SEV3", "SEV4"}
STATUSES = {"open", "in_progress", "closed"}
GROUPS = {"severity", "status", "category", "tier", "region"}
SLA = {
    "Basic": {"first_response_critical": "8 business hours", "uptime": "99.5%"},
    "Business": {"first_response_critical": "1 hour", "uptime": "99.9%"},
    "Enterprise": {"first_response_critical": "15 minutes", "uptime": "99.95%"},
}
MAX_OBS_CHARS = 1200


# ---------------------------------------------------------------- BM25 over runbooks
STOP = frozenset("a an the and or of to in on for with by at from is are was were be been it its this that these those "
                 "as if then than so but not no do does did how what when which who why can could should would will may must "
                 "i we you they our your their there here about into over per need needed involved says say reports".split())


def _tok(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOP]


class RunbookIndex:
    def __init__(self, runbook_dir: Path, max_chars: int = 500):
        self.passages: list[tuple[str, str]] = []   # (source, text)
        for path in sorted(Path(runbook_dir).glob("*.md")):
            title, section, buf = path.stem, "", []

            def flush():
                body = " ".join(buf).strip()
                if body:
                    self.passages.append((path.stem, f"{title} / {section}: {body}"[:max_chars * 2]))

            for line in path.read_text().splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## "):
                    flush()
                    section, buf = line[3:].strip(), []
                elif line.strip():
                    buf.append(line.strip())
            flush()
        self._docs = [_tok(t) for _, t in self.passages]
        self._avg = sum(map(len, self._docs)) / max(1, len(self._docs))
        self._df: Counter = Counter(t for d in self._docs for t in set(d))

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        n, scores = len(self._docs), defaultdict(float)
        for term in set(_tok(query)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self._docs):
                tf = doc.count(term)
                if tf:
                    scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(doc) / self._avg))
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.passages[i][0], self.passages[i][1], round(s, 2)) for i, s in top]


# ---------------------------------------------------------------- tool plumbing
@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, type]                  # required params
    optional: dict[str, type] = field(default_factory=dict)
    fn: Callable[..., str] = lambda **_: ""

    def spec(self) -> str:
        req = ", ".join(f"{k}: {t.__name__}" for k, t in self.params.items())
        opt = ", ".join(f"{k}?: {t.__name__}" for k, t in self.optional.items())
        return f"{self.name}({', '.join(x for x in (req, opt) if x)}) - {self.description}"

    def call(self, args: dict) -> str:
        if not isinstance(args, dict):
            return "ERROR: action_input must be a JSON object"
        unknown = set(args) - set(self.params) - set(self.optional)
        if unknown:
            return f"ERROR: unknown argument(s) {sorted(unknown)}. Usage: {self.spec()}"
        for k, t in self.params.items():
            if k not in args:
                return f"ERROR: missing required argument '{k}'. Usage: {self.spec()}"
        for k, v in args.items():
            t = {**self.params, **self.optional}[k]
            if not isinstance(v, t) or (t is int and isinstance(v, bool)):
                return f"ERROR: argument '{k}' must be {t.__name__}, got {type(v).__name__}. Usage: {self.spec()}"
        try:
            return self.fn(**args)[:MAX_OBS_CHARS]
        except Exception as exc:  # a tool crash is an observation, not a process crash
            return f"ERROR: {type(exc).__name__}: {exc}"


class Toolbox:
    def __init__(self, db_path: Path, runbook_dir: Path, state_dir: Path):
        self.db_path, self.state_dir = Path(db_path), Path(state_dir)
        self.index = RunbookIndex(runbook_dir)
        self.tools: dict[str, Tool] = {}
        self._register()

    # -- db helper: read-only connection (mode=ro) so even a bug cannot write
    def _q(self, sql: str, params: tuple = ()) -> list[tuple]:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def _register(self):
        def add(tool: Tool):
            self.tools[tool.name] = tool

        def search_runbooks(query: str, k: int = 3) -> str:
            hits = self.index.search(query, max(1, min(k, 5)))
            if not hits:
                return "NO_RESULTS"
            return "\n".join(f"[{src}] {text}" for src, text, _ in hits)

        def get_customer(name: str) -> str:
            rows = self._q("SELECT id, name, tier, region FROM customers WHERE lower(name) = lower(?)", (name.strip(),))
            if not rows:
                rows = self._q("SELECT id, name, tier, region FROM customers WHERE lower(name) LIKE lower(?) LIMIT 3",
                               (f"%{name.strip()}%",))
            if not rows:
                return f"NOT_FOUND: no customer matching '{name}'"
            return "\n".join(f"customer_id={r[0]} name={r[1]} tier={r[2]} region={r[3]}" for r in rows)

        def list_tickets(customer: str = "", status: str = "", severity: str = "", category: str = "", limit: int = 5) -> str:
            if status and status not in STATUSES:
                return f"ERROR: status must be one of {sorted(STATUSES)}"
            if severity and severity not in SEVERITIES:
                return f"ERROR: severity must be one of {sorted(SEVERITIES)}"
            where, params = [], []
            for col, val in (("c.name", customer), ("t.status", status), ("t.severity", severity), ("t.category", category)):
                if val:
                    where.append(f"lower({col}) = lower(?)")
                    params.append(val)
            sql = ("SELECT t.id, c.name, c.tier, t.severity, t.status, t.category, t.description FROM tickets t "
                   "JOIN customers c ON c.id = t.customer_id"
                   + (" WHERE " + " AND ".join(where) if where else "")
                   + " ORDER BY t.severity, t.id DESC LIMIT ?")
            rows = self._q(sql, (*params, max(1, min(limit, 10))))
            if not rows:
                return "NO_TICKETS"
            return "\n".join(f"T-{r[0]} {r[3]} {r[4]} [{r[5]}] {r[1]} ({r[2]}): {r[6]}" for r in rows)

        def ticket_stats(group_by: str, status: str = "", severity: str = "") -> str:
            if group_by not in GROUPS:
                return f"ERROR: group_by must be one of {sorted(GROUPS)}"
            col = {"tier": "c.tier", "region": "c.region"}.get(group_by, f"t.{group_by}")  # allow-listed, never user text
            where, params = [], []
            if status:
                where.append("t.status = ?"); params.append(status)
            if severity:
                where.append("t.severity = ?"); params.append(severity)
            sql = (f"SELECT {col}, COUNT(*) FROM tickets t JOIN customers c ON c.id = t.customer_id"
                   + (" WHERE " + " AND ".join(where) if where else "") + f" GROUP BY {col} ORDER BY 2 DESC")
            rows = self._q(sql, tuple(params))
            return ", ".join(f"{k}={v}" for k, v in rows) or "NO_ROWS"

        def get_sla(tier: str) -> str:
            sla = SLA.get(tier.strip().capitalize())
            return f"{tier.capitalize()}: first response (critical) {sla['first_response_critical']}, uptime {sla['uptime']}" if sla \
                else f"ERROR: unknown tier '{tier}'. Valid: {sorted(SLA)}"

        add(Tool("search_runbooks", "search internal runbooks and policies", {"query": str}, {"k": int}, search_runbooks))
        add(Tool("get_sla", "SLA commitments for a support tier (Basic|Business|Enterprise)", {"tier": str}, fn=get_sla))
        add(Tool("get_customer", "look up a customer's tier and region by name", {"name": str}, fn=get_customer))
        add(Tool("list_tickets", "list tickets, optionally filtered", {},
                 {"customer": str, "status": str, "severity": str, "category": str, "limit": int}, list_tickets))
        add(Tool("ticket_stats", "count tickets grouped by severity|status|category|tier|region", {"group_by": str},
                 {"status": str, "severity": str}, ticket_stats))

    # -- the one side-effecting tool, callable only by the approval gate
    def create_escalation(self, ticket_id: int, team: str, summary: str) -> str:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        record = {"ticket_id": ticket_id, "team": team, "summary": summary[:500]}
        with (self.state_dir / "escalations.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return f"escalation created: ticket {ticket_id} -> {team}"

    def specs(self, names: list[str]) -> str:
        return "\n".join(self.tools[n].spec() for n in names)
