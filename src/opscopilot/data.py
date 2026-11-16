"""Seed a small deterministic ticket database (all data is fictional)."""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

PREFIX = ["Nova", "Apex", "Blue", "Cedar", "Delta", "Echo", "Fjord", "Gamma", "Helix", "Iris"]
SUFFIX = ["Logistics", "Retail", "Foods", "Motors", "Pharma", "Traders"]
REGIONS = ["EMEA", "APAC", "NA", "LATAM"]
TIERS = ["Basic", "Business", "Enterprise"]
TEMPLATES = {
    "api": "Receiving HTTP 429 Too Many Requests on {ep} after a traffic spike.",
    "billing": "Duplicate charge of {amt} USD on invoice INV-{inv}.",
    "outage": "Tracking dashboard unavailable in {reg}, requests return HTTP 503.",
    "security": "Staff member clicked a phishing link and entered credentials.",
    "access": "User locked out of the dashboard after changing phones, MFA device lost.",
    "delivery": "Parcel MF{trk} has no tracking updates for {d} days.",
}

SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, tier TEXT NOT NULL, region TEXT NOT NULL);
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    severity TEXT NOT NULL, status TEXT NOT NULL, category TEXT NOT NULL,
    description TEXT NOT NULL, opened_at TEXT NOT NULL
);
CREATE INDEX idx_t_customer ON tickets(customer_id);
"""


def build_db(path: Path, seed: int = 7, n_customers: int = 24, n_tickets: int = 90) -> Path:
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    names = set()
    while len(names) < n_customers:
        names.add(f"{rng.choice(PREFIX)} {rng.choice(SUFFIX)}")
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                    [(i, n, rng.choices(TIERS, [4, 3, 3])[0], rng.choice(REGIONS))
                     for i, n in enumerate(sorted(names), 1)])
    rows = []
    for i in range(1, n_tickets + 1):
        cat = rng.choices(list(TEMPLATES), [25, 15, 12, 8, 15, 25])[0]
        sev = "SEV1" if cat == "outage" and rng.random() < 0.5 else (
            "SEV2" if cat == "security" else rng.choices(["SEV2", "SEV3", "SEV4"], [10, 45, 45])[0])
        desc = TEMPLATES[cat].format(ep=rng.choice(["/v1/shipments", "/v1/rates", "/v1/tracking"]),
                                     amt=rng.choice([49, 99, 249]), inv=rng.randint(1000, 9999),
                                     reg=rng.choice(REGIONS), trk=rng.randint(10**8, 10**9 - 1), d=rng.randint(4, 12))
        rows.append((i, rng.randint(1, n_customers), sev, rng.choices(["open", "in_progress", "closed"], [30, 20, 50])[0],
                     cat, desc, (datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 300))).isoformat(timespec="minutes")))
    con.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path
