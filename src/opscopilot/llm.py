"""LLM backends. Every agent talks to the model through one method and always gets JSON text back.

  GeminiLLM     - Gemini via Vertex AI or API key (google-genai SDK), JSON mime type, usage metadata.
  RuleBasedLLM  - a deterministic stand-in policy for each role. It exists so the graph, tools,
                  guardrails, checkpointing and tests can run offline in milliseconds. It does NOT reason:
                  it follows small hand-written rules, so results with it say nothing about LLM quality.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings

CATEGORY_HINTS = [("api", r"\b429|rate limit|too many requests\b"), ("security", r"phish|credential|breach"),
                  ("outage", r"\b50[23]\b|outage|dashboard (is )?(down|unavailable)"), ("billing", r"refund|charge|invoice|billing"),
                  ("access", r"locked out|mfa|login"), ("delivery", r"parcel|tracking|delivery|shipment is late")]
# query expansion: the offline policy's stand-in for an LLM rewriting "dashboard is down" into runbook vocabulary
EXPANSION = {"api": "rate limit backoff retry-after limits", "security": "phishing credentials rotate report",
             "outage": "outage incident severity acknowledge escalation failover communication",
             "billing": "refund approval duplicate charge", "access": "lockout MFA identity", "delivery": "tracking parcel lost"}


def task_category(task: str) -> str:
    return next((c for c, rx in CATEGORY_HINTS if re.search(rx, task, re.I)), "")


CUSTOMER_RE = re.compile(r"\b([A-Z][a-z]+ (?:Logistics|Retail|Foods|Motors|Pharma|Traders|Corp|Inc|Ltd|Group))\b")
TIER_RE = re.compile(r"\b(Basic|Business|Enterprise)\b")


@dataclass
class LLMRequest:
    role: str
    system: str
    prompt: str
    view: dict = field(default_factory=dict)   # structured context; only RuleBasedLLM reads it


@dataclass
class LLMReply:
    text: str
    in_tokens: int
    out_tokens: int
    ms: float


class LLM(Protocol):
    def complete(self, req: LLMRequest) -> LLMReply: ...


def parse_json(text: str) -> dict:
    """Parse a model reply into a dict. Tolerates ```json fences and prose around the object."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise ValueError("no JSON object found in model reply")
        out = json.loads(m.group(0))
    if not isinstance(out, dict):
        raise ValueError("model reply is not a JSON object")
    return out


# ------------------------------------------------------------------ Gemini
class GeminiLLM:
    def __init__(self, settings: Settings):
        from google import genai

        self._client = (genai.Client(vertexai=True, project=settings.gcp_project, location=settings.gcp_location)
                        if settings.use_vertex else genai.Client())
        self._model = settings.gemini_model

    def complete(self, req: LLMRequest) -> LLMReply:
        from google.genai import types

        t = time.perf_counter()
        res = self._client.models.generate_content(
            model=self._model, contents=req.prompt,
            config=types.GenerateContentConfig(system_instruction=req.system, temperature=0.2,
                                               response_mime_type="application/json", max_output_tokens=1200))
        u = getattr(res, "usage_metadata", None)
        text = res.text or ""
        return LLMReply(text, getattr(u, "prompt_token_count", None) or (len(req.system) + len(req.prompt)) // 4,
                        getattr(u, "candidates_token_count", None) or len(text) // 4, (time.perf_counter() - t) * 1000)


# ------------------------------------------------------------------ offline policy
class RuleBasedLLM:
    def complete(self, req: LLMRequest) -> LLMReply:
        t = time.perf_counter()
        out = getattr(self, f"_{req.role}")(req.view)
        text = json.dumps(out)
        return LLMReply(text, (len(req.system) + len(req.prompt)) // 4, len(text) // 4, (time.perf_counter() - t) * 1000)

    # -- supervisor: decide which workers are needed, then hand over to the drafter
    def _supervisor(self, v: dict) -> dict:
        task, findings = v["task"], v.get("findings", {})
        if not v.get("plan"):
            plan = ["researcher"]
            if CUSTOMER_RE.search(task) or re.search(r"\b(how many|tickets?|customers?|sev[1-4]|open)\b", task, re.I):
                plan.append("analyst")
            return {"thought": "Split the task between knowledge search and data lookup.", "plan": plan, "dispatch": plan}
        missing = [w for w in v["plan"] if w not in findings]
        if missing:
            return {"thought": "Some workers have not reported yet.", "plan": v["plan"], "dispatch": missing}
        return {"thought": "All findings are in; write the response.", "plan": v["plan"], "dispatch": ["drafter"]}

    # -- researcher: runbook search, then answer
    def _researcher(self, v: dict) -> dict:
        scratch, task = v["scratch"], v["task"]
        if not scratch:
            return {"thought": "Search the runbooks, rewriting the task into runbook vocabulary.", "action": "search_runbooks",
                    "action_input": {"query": f"{task} {EXPANSION.get(task_category(task), '')}".strip(), "k": 3}}
        obs = scratch[0]["observation"]
        sources = sorted(set(re.findall(r"^\[([a-z_]+)\]", obs, re.M)))
        return {"thought": "I have enough guidance.", "action": "final",
                "answer": obs if not obs.startswith("NO_RESULTS") else "No relevant runbook found.", "sources": sources}

    # -- analyst: customer -> SLA -> tickets -> stats
    def _analyst(self, v: dict) -> dict:
        scratch, task = v["scratch"], v["task"]
        m = CUSTOMER_RE.search(task)
        wants_stats = bool(re.search(r"\b(how many|count|number of)\b", task, re.I))
        steps: list[tuple[str, dict]] = []
        if m:
            cat = task_category(task)
            steps += [("get_customer", {"name": m.group(1)}), ("get_sla", {"tier": "__tier__"}),
                      ("list_tickets", {"customer": m.group(1), "limit": 5, **({"category": cat} if cat else {})})]
        if wants_stats or not m:
            kw = {"group_by": "severity"}
            if re.search(r"\bopen\b", task, re.I):
                kw["status"] = "open"
            if (s := re.search(r"\bSEV[1-4]\b", task, re.I)):
                kw = {"group_by": "tier", "severity": s.group(0).upper(), **({"status": "open"} if "status" in kw else {})}
            steps.append(("ticket_stats", kw))
        n = len(scratch)
        if scratch and scratch[0]["action"] == "get_customer" and scratch[0]["observation"].startswith("NOT_FOUND"):
            return {"thought": "Customer does not exist; stop instead of guessing.", "action": "final",
                    "answer": f"Customer '{m.group(1)}' was not found in the customer database.", "sources": []}
        if n < len(steps):
            name, args = steps[n]
            if args.get("tier") == "__tier__":
                tm = TIER_RE.search(scratch[0]["observation"]) if scratch else None
                if not tm:
                    return {"thought": "Tier unknown, skip SLA.", "action": "list_tickets", "action_input": steps[2][1]}
                args = {"tier": tm.group(1)}
            return {"thought": f"Next: {name}.", "action": name, "action_input": args}
        return {"thought": "Data gathered.", "action": "final",
                "answer": "\n".join(s["observation"] for s in scratch) or "No data.", "sources": ["tickets_db"]}

    # -- drafter: first pass is deliberately terse; revisions address the critique
    def _drafter(self, v: dict) -> dict:
        f, task, critique = v["findings"], v["task"], v.get("critique") or []
        res = f.get("researcher", {}).get("answer", "")
        ana = f.get("analyst", {}).get("answer", "")
        sources = sorted({s for w in f.values() for s in w.get("sources", [])})
        tickets = re.findall(r"T-(\d+) (SEV[1-4]) (open|in_progress|closed) \[(\w+)\]", ana)
        sla = re.search(r"(Basic|Business|Enterprise): first response \(critical\) ([^,]+), uptime ([\d.]+%)", ana)
        # keep the sentences that overlap most with the task (original order), not simply the first ones
        from .tools import _tok
        task_terms = set(_tok(f"{task} {EXPANSION.get(task_category(task), '')}"))
        cands = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[[a-z_]+\][^:]*: ", "", res)) if len(s.strip()) > 25]
        ranked = sorted(range(len(cands)), key=lambda i: (-len(task_terms & set(_tok(cands[i]))), i))[:4]
        guidance = [cands[i] for i in sorted(ranked) if task_terms & set(_tok(cands[i]))]

        lines = ["## Summary", task.strip().rstrip(".?!") + "."]
        if ana and "was not found" in ana:
            lines.append(ana)
        elif tickets:
            lines.append(f"Found {len(tickets)} relevant ticket(s): " + ", ".join(f"T-{i} ({s}, {st}, {c})" for i, s, st, c in tickets[:3]) + ".")
        elif ana:
            lines.append("Data lookup: " + ana.splitlines()[0])
        lines += ["", "## Severity and SLA"]
        lines.append(f"{sla.group(1)} plan: first response for critical issues within {sla.group(2)}, uptime commitment {sla.group(3)}."
                     if sla else "No SLA information available for this request.")
        lines += ["", "## Next steps"] + [f"{i}. {g}" for i, g in enumerate(guidance or ["Escalate to the support lead for manual review."], 1)]
        issues = " ".join(critique).lower()
        if "root cause" in issues:
            cats = {c for *_, c in tickets}
            hint = ("rate limiting after a traffic spike" if "api" in cats else "a regional service failure" if "outage" in cats
                    else "compromised credentials" if "security" in cats else "not determinable from the available data")
            lines += ["", "## Likely root cause", f"Most likely {hint}; to be confirmed with the on-call engineer."]
        if "customer message" in issues:
            lines += ["", "## Customer message",
                      "Hello, thanks for reaching out. We are looking into this now and will update you with the next steps shortly. "
                      "We will not guess at a fix time until engineering confirms it."]
        if "sources" in issues:
            lines += ["", "## Sources", ", ".join(sources) if sources else "none"]
        esc = None
        for tid, sev, st, cat in tickets:
            if st != "closed" and (sev in ("SEV1", "SEV2") or cat == "security"):
                team = {"outage": "Platform on-call", "security": "Security Operations", "billing": "Finance Operations"}.get(cat, "Support Lead")
                esc = {"ticket_id": int(tid), "team": team, "summary": f"{sev} {cat} ticket needs attention: {task[:200]}"}
                break
        return {"draft": "\n".join(lines), "escalation": esc}

    # -- critic: judges the draft against the rubric (deterministic checks run in code as well)
    def _critic(self, v: dict) -> dict:
        from .agents import rubric_issues

        issues = rubric_issues(v["draft"])
        return {"ok": not issues, "issues": issues}
