"""Per-customer API keys for the gateway.

The gateway's original auth was a single shared bearer token (``api_token`` in
:func:`repercep.serving.app.create_app`) — fine for one design partner on one
box, useless the moment two customers share a deployment: you cannot attribute
a request, cannot revoke one caller without rotating everybody, and cannot
bill. This module replaces it with per-customer keys while leaving the shared
token working, so existing deployments are unaffected.

Storage is SQLite, one file, no server. That is a deliberate scale ceiling
(see :class:`KeyStore`) chosen because the alternative — standing up Postgres
— is P1 work and this is the P0 slice: enough tenancy to hand a design partner
a key and invoice them accurately.

**Keys are stored hashed.** A leaked database file does not yield working
credentials. The plaintext key is returned exactly once, at mint time, and is
unrecoverable afterwards — the same contract every cloud provider offers, and
the reason :func:`mint_key` is the only place a full key ever exists.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

# Key wire format: ``rpc_<key_id>_<secret>``.
#
# The key_id travels *in* the key so a presented credential can be looked up
# with one indexed read rather than by hashing against every stored row, and
# so logs/support tickets can name a key ("rpc_a1b2c3d4…") without ever
# handling the secret half.
_KEY_PREFIX: Final = "rpc"
_KEY_ID_BYTES: Final = 4  # → 8 hex chars
_SECRET_BYTES: Final = 32  # → 43 urlsafe-base64 chars

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id              TEXT PRIMARY KEY,
    key_hash            TEXT NOT NULL,
    customer            TEXT NOT NULL,
    label               TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    revoked_at          TEXT,
    monthly_token_quota INTEGER
);
CREATE INDEX IF NOT EXISTS idx_api_keys_customer ON api_keys(customer);
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def hash_key(full_key: str) -> str:
    """Hash a full API key for storage/comparison.

    Plain SHA-256, deliberately *not* a slow KDF: an API key is 32 bytes of
    CSPRNG output, not a human-chosen password, so it has no dictionary to
    attack and stretching would only add per-request latency to the auth path.
    """
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


def mint_key() -> tuple[str, str]:
    """Generate a new key. Returns ``(full_key, key_id)``.

    The full key is shown once by the caller and never stored; only
    :func:`hash_key` of it goes to the database.
    """
    key_id = secrets.token_hex(_KEY_ID_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    return f"{_KEY_PREFIX}_{key_id}_{secret}", key_id


def parse_key_id(full_key: str) -> str | None:
    """Extract the key_id from a presented key, or ``None`` if malformed.

    ``maxsplit=2`` matters: ``token_urlsafe`` emits ``_`` inside the secret,
    so a plain ``split("_")`` would shred it.
    """
    parts = full_key.split("_", 2)
    if len(parts) != 3 or parts[0] != _KEY_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1]


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller behind a request.

    ``key_id`` is the attribution unit for usage; ``customer`` is the billing
    unit. One customer may hold several keys (staging vs production, or one
    per service) and they roll up to one invoice.
    """

    key_id: str
    customer: str
    label: str = ""
    monthly_token_quota: int | None = None

    @property
    def is_legacy_shared_token(self) -> bool:
        """True for the pre-tenancy single shared token (see :mod:`app`)."""
        return self.key_id == LEGACY_KEY_ID


#: Attribution for callers authenticating with the old shared ``api_token``.
#: Their usage is still metered — it simply cannot be split per customer,
#: which is the whole reason per-customer keys exist.
LEGACY_KEY_ID: Final = "legacy-shared"


@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    """A stored key, without the secret."""

    key_id: str
    customer: str
    label: str
    created_at: str
    revoked_at: str | None
    monthly_token_quota: int | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class KeyStore:
    """SQLite-backed API key store.

    Thread-safe via a coarse lock around a single ``check_same_thread=False``
    connection. That is correct but serializes writes, which is acceptable
    because every operation here is a sub-millisecond indexed read/write and
    the auth path does one of them per request.

    **Scale ceiling, stated so it is not discovered in production:** one
    SQLite file on one box. It does not survive the gateway becoming more
    than one process on more than one machine. The replacement is Postgres,
    and it is P1 work — this exists so P0 can bill a design partner correctly,
    not so it can run a public cloud.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if self._path.parent != Path(""):
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL so a concurrent reader (the usage ledger's own connection to the
        # same file) is never blocked by a writer.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- mutation -----------------------------------------------------------

    def create(
        self,
        *,
        customer: str,
        label: str = "",
        monthly_token_quota: int | None = None,
    ) -> tuple[str, ApiKeyRecord]:
        """Mint and store a key. Returns ``(full_key, record)``.

        The full key is the *only* time the plaintext exists. Surface it to
        the operator immediately and do not log it.
        """
        if not customer.strip():
            raise ValueError("customer must be a non-empty name")
        full_key, key_id = mint_key()
        created = _utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_keys"
                " (key_id, key_hash, customer, label, created_at, monthly_token_quota)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (key_id, hash_key(full_key), customer, label, created, monthly_token_quota),
            )
            self._conn.commit()
        return full_key, ApiKeyRecord(
            key_id=key_id,
            customer=customer,
            label=label,
            created_at=created,
            revoked_at=None,
            monthly_token_quota=monthly_token_quota,
        )

    def revoke(self, key_id: str) -> bool:
        """Revoke a key. Returns False if unknown or already revoked.

        Revocation is a tombstone, not a delete: the usage rows attributed to
        this key must stay explicable at invoice time.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (_utcnow(), key_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # -- lookup -------------------------------------------------------------

    def authenticate(self, full_key: str) -> Principal | None:
        """Resolve a presented key to a :class:`Principal`, or ``None``.

        Returns ``None`` for malformed, unknown, revoked, and wrong-secret
        keys alike — the caller must not be able to tell those apart.
        """
        key_id = parse_key_id(full_key)
        if key_id is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
        if row is None or row["revoked_at"] is not None:
            return None
        # compare_digest: the stored value is a hash, but constant-time
        # comparison costs nothing and removes the question entirely.
        if not hmac.compare_digest(row["key_hash"], hash_key(full_key)):
            return None
        return Principal(
            key_id=row["key_id"],
            customer=row["customer"],
            label=row["label"],
            monthly_token_quota=row["monthly_token_quota"],
        )

    def get(self, key_id: str) -> ApiKeyRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
        return None if row is None else _record(row)

    def list_keys(self, *, customer: str | None = None, include_revoked: bool = True) -> list[
        ApiKeyRecord
    ]:
        sql = "SELECT * FROM api_keys"
        clauses: list[str] = []
        params: list[object] = []
        if customer is not None:
            clauses.append("customer = ?")
            params.append(customer)
        if not include_revoked:
            clauses.append("revoked_at IS NULL")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, key_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_record(r) for r in rows]


def _record(row: sqlite3.Row) -> ApiKeyRecord:
    return ApiKeyRecord(
        key_id=row["key_id"],
        customer=row["customer"],
        label=row["label"],
        created_at=row["created_at"],
        revoked_at=row["revoked_at"],
        monthly_token_quota=row["monthly_token_quota"],
    )


__all__ = [
    "LEGACY_KEY_ID",
    "ApiKeyRecord",
    "KeyStore",
    "Principal",
    "hash_key",
    "mint_key",
    "parse_key_id",
]
