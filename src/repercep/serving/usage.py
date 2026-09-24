"""Append-only usage ledger — the thing you invoice from.

One row per billable call, written after the response completes so the token
counts are the real ones rather than a prediction. Rows are never updated or
deleted: an invoice must stay reconstructible, and a mutable usage table is
how billing disputes become unwinnable.

**The ``exact`` column is the honest bit.** Token counts come from the
upstream engine's own ``usage`` object wherever it can be obtained, and that
is the overwhelmingly common case (see :mod:`repercep.serving.metering`).
Where it cannot — a client streaming against an upstream that omits usage,
or a call that failed mid-stream — the row is written with ``exact = 0`` and
an estimate. Billing on estimated rows without saying so is the kind of thing
that ends a design-partner relationship, so the column exists, the CLI prints
it, and :meth:`UsageLedger.summary` reports estimated tokens separately.

Storage is the same SQLite file as :mod:`repercep.serving.tenancy`, with the
same stated scale ceiling.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,
    key_id            TEXT    NOT NULL,
    customer          TEXT    NOT NULL,
    route             TEXT    NOT NULL,
    model             TEXT    NOT NULL DEFAULT '',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    exact             INTEGER NOT NULL DEFAULT 1,
    status            INTEGER NOT NULL DEFAULT 200,
    latency_ms        REAL    NOT NULL DEFAULT 0.0,
    request_id        TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_usage_key_ts      ON usage_events(key_id, ts);
CREATE INDEX IF NOT EXISTS idx_usage_customer_ts ON usage_events(customer, ts);
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def month_start(now: datetime | None = None) -> str:
    """ISO timestamp of the start of the current UTC month.

    The quota window. UTC rather than local time so a deployment that moves
    region does not silently re-cut its billing periods.
    """
    ref = now or datetime.now(UTC)
    return ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(
        timespec="seconds"
    )


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One billable call."""

    key_id: str
    customer: str
    route: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    exact: bool = True
    status: int = 200
    latency_ms: float = 0.0
    request_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Rolled-up usage for one key or customer over a window."""

    key_id: str
    customer: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_tokens: int
    """Tokens from rows where ``exact = 0``. A subset of the totals above,
    surfaced separately so an invoice can disclose its own uncertainty."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageLedger:
    """Append-only usage store over SQLite.

    Opens its own connection to the same file as
    :class:`~repercep.serving.tenancy.KeyStore`; WAL mode (set there and here)
    keeps the two from blocking each other.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if self._path.parent != Path(""):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- write --------------------------------------------------------------

    def record(self, event: UsageEvent) -> None:
        """Append one usage row. Never raises on a well-formed event."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_events"
                " (ts, key_id, customer, route, model, prompt_tokens, completion_tokens,"
                "  exact, status, latency_ms, request_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _utcnow(),
                    event.key_id,
                    event.customer,
                    event.route,
                    event.model,
                    event.prompt_tokens,
                    event.completion_tokens,
                    int(event.exact),
                    event.status,
                    event.latency_ms,
                    event.request_id,
                ),
            )
            self._conn.commit()

    # -- read ---------------------------------------------------------------

    def tokens_since(self, *, key_id: str | None = None, customer: str | None = None,
                     since: str) -> int:
        """Total tokens billed to a key or customer since an ISO timestamp.

        The quota check. Exactly one of ``key_id``/``customer`` must be given.
        """
        if (key_id is None) == (customer is None):
            raise ValueError("pass exactly one of key_id or customer")
        column = "key_id" if key_id is not None else "customer"
        value = key_id if key_id is not None else customer
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS t"
                f" FROM usage_events WHERE {column} = ? AND ts >= ?",
                (value, since),
            ).fetchone()
        return int(row["t"])

    def summary(self, *, since: str | None = None, customer: str | None = None) -> list[
        UsageSummary
    ]:
        """Per-key rollup, newest-heaviest first. The invoice query."""
        sql = (
            "SELECT key_id, customer,"
            "       COUNT(*) AS calls,"
            "       COALESCE(SUM(prompt_tokens), 0) AS pt,"
            "       COALESCE(SUM(completion_tokens), 0) AS ct,"
            "       COALESCE(SUM(CASE WHEN exact = 0"
            "                    THEN prompt_tokens + completion_tokens ELSE 0 END), 0) AS est"
            " FROM usage_events"
        )
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if customer is not None:
            clauses.append("customer = ?")
            params.append(customer)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY key_id, customer ORDER BY pt + ct DESC, key_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            UsageSummary(
                key_id=r["key_id"],
                customer=r["customer"],
                calls=int(r["calls"]),
                prompt_tokens=int(r["pt"]),
                completion_tokens=int(r["ct"]),
                estimated_tokens=int(r["est"]),
            )
            for r in rows
        ]

    def recent(self, *, key_id: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        """Raw recent rows — for support questions and reconciliation."""
        sql = "SELECT * FROM usage_events"
        params: list[object] = []
        if key_id is not None:
            sql += " WHERE key_id = ?"
            params.append(key_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())


__all__ = ["UsageEvent", "UsageLedger", "UsageSummary", "month_start"]
