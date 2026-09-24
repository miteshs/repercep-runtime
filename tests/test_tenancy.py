"""Tests for per-customer API keys and the usage ledger.

Both stores are plain SQLite over a tmp_path file — no server, no GPU, no
network. What is asserted here is the part that has to be right before a
customer is billed from it: keys are unrecoverable after minting, revocation
is immediate and does not erase history, and usage rolls up per key and per
customer with estimated tokens reported separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from repercep.serving.tenancy import (
    KeyStore,
    hash_key,
    mint_key,
    parse_key_id,
)
from repercep.serving.usage import UsageEvent, UsageLedger, month_start

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------


def test_mint_key_format_and_id_roundtrip() -> None:
    full_key, key_id = mint_key()
    assert full_key.startswith("rpc_")
    assert parse_key_id(full_key) == key_id


def test_parse_key_id_rejects_malformed() -> None:
    for bad in ("", "rpc", "rpc_", "rpc_abc", "nope_abc_def", "rpc__secret", "rpc_abc_"):
        assert parse_key_id(bad) is None


def test_secret_may_contain_underscores() -> None:
    """``token_urlsafe`` emits ``_``; a naive split would shred the secret."""
    assert parse_key_id("rpc_deadbeef_aa_bb_cc") == "deadbeef"


def test_keys_are_unique() -> None:
    assert len({mint_key()[0] for _ in range(200)}) == 200


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_create_and_authenticate(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "gw.db")
    full_key, record = store.create(customer="acme", label="prod")

    principal = store.authenticate(full_key)
    assert principal is not None
    assert principal.key_id == record.key_id
    assert principal.customer == "acme"
    assert principal.label == "prod"
    store.close()


def test_plaintext_key_is_not_stored(tmp_path: Path) -> None:
    """The DB file must not contain the credential, only its hash."""
    db = tmp_path / "gw.db"
    store = KeyStore(db)
    full_key, _ = store.create(customer="acme")
    store.close()

    blob = db.read_bytes()
    assert full_key.encode() not in blob
    assert hash_key(full_key).encode() in blob


def test_authenticate_rejects_unknown_and_wrong_secret(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "gw.db")
    full_key, record = store.create(customer="acme")

    assert store.authenticate("rpc_00000000_nope") is None
    assert store.authenticate("not-a-key") is None
    # Right key_id, wrong secret — the one that matters.
    assert store.authenticate(f"rpc_{record.key_id}_wrongsecret") is None
    assert store.authenticate(full_key) is not None
    store.close()


def test_revoke_is_immediate_and_idempotent(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "gw.db")
    full_key, record = store.create(customer="acme")

    assert store.revoke(record.key_id) is True
    assert store.authenticate(full_key) is None
    # Second revoke is a no-op, not an error.
    assert store.revoke(record.key_id) is False
    # The row survives as a tombstone so past usage stays explicable.
    stored = store.get(record.key_id)
    assert stored is not None
    assert stored.active is False
    assert stored.revoked_at is not None
    store.close()


def test_revoking_one_key_leaves_siblings_working(tmp_path: Path) -> None:
    """The whole point of per-customer keys over one shared token."""
    store = KeyStore(tmp_path / "gw.db")
    key_a, rec_a = store.create(customer="acme", label="staging")
    key_b, _ = store.create(customer="acme", label="prod")

    store.revoke(rec_a.key_id)
    assert store.authenticate(key_a) is None
    assert store.authenticate(key_b) is not None
    store.close()


def test_list_keys_filters(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "gw.db")
    _, rec_a = store.create(customer="acme")
    store.create(customer="acme")
    store.create(customer="globex")
    store.revoke(rec_a.key_id)

    assert len(store.list_keys()) == 3
    assert len(store.list_keys(customer="acme")) == 2
    assert len(store.list_keys(customer="acme", include_revoked=False)) == 1
    assert len(store.list_keys(include_revoked=False)) == 2
    store.close()


def test_store_survives_reopen(tmp_path: Path) -> None:
    db = tmp_path / "gw.db"
    store = KeyStore(db)
    full_key, _ = store.create(customer="acme")
    store.close()

    reopened = KeyStore(db)
    assert reopened.authenticate(full_key) is not None
    reopened.close()


def test_quota_travels_on_the_principal(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "gw.db")
    full_key, _ = store.create(customer="acme", monthly_token_quota=1_000)
    principal = store.authenticate(full_key)
    assert principal is not None
    assert principal.monthly_token_quota == 1_000
    store.close()


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------


def _event(key_id: str = "k1", customer: str = "acme", **kw: object) -> UsageEvent:
    base: dict[str, object] = {
        "key_id": key_id,
        "customer": customer,
        "route": "/v1/chat/completions",
        "model": "qwen2.5-7b",
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }
    base.update(kw)
    return UsageEvent(**base)  # type: ignore[arg-type]


def test_summary_rolls_up_per_key(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "gw.db")
    ledger.record(_event())
    ledger.record(_event())
    ledger.record(_event(key_id="k2", customer="globex", prompt_tokens=7, completion_tokens=3))

    rows = {r.key_id: r for r in ledger.summary()}
    assert rows["k1"].calls == 2
    assert rows["k1"].prompt_tokens == 200
    assert rows["k1"].completion_tokens == 100
    assert rows["k1"].total_tokens == 300
    assert rows["k2"].total_tokens == 10
    ledger.close()


def test_summary_filters_by_customer(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "gw.db")
    ledger.record(_event())
    ledger.record(_event(key_id="k2", customer="globex"))

    rows = ledger.summary(customer="globex")
    assert len(rows) == 1
    assert rows[0].customer == "globex"
    ledger.close()


def test_estimated_tokens_reported_separately(tmp_path: Path) -> None:
    """An invoice has to be able to disclose its own uncertainty."""
    ledger = UsageLedger(tmp_path / "gw.db")
    ledger.record(_event())  # exact
    ledger.record(_event(exact=False, prompt_tokens=0, completion_tokens=42))

    row = ledger.summary()[0]
    assert row.total_tokens == 192
    assert row.estimated_tokens == 42
    ledger.close()


def test_tokens_since_counts_the_quota_window(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "gw.db")
    ledger.record(_event())
    ledger.record(_event())

    assert ledger.tokens_since(key_id="k1", since=month_start()) == 300
    assert ledger.tokens_since(customer="acme", since=month_start()) == 300
    assert ledger.tokens_since(key_id="k1", since="2999-01-01T00:00:00+00:00") == 0
    ledger.close()


def test_tokens_since_requires_exactly_one_selector(tmp_path: Path) -> None:
    ledger = UsageLedger(tmp_path / "gw.db")
    for kwargs in ({}, {"key_id": "k1", "customer": "acme"}):
        try:
            ledger.tokens_since(since=month_start(), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
    ledger.close()


def test_ledger_is_append_only_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "gw.db"
    ledger = UsageLedger(db)
    ledger.record(_event())
    ledger.close()

    reopened = UsageLedger(db)
    reopened.record(_event())
    assert reopened.summary()[0].calls == 2
    assert len(reopened.recent()) == 2
    reopened.close()


def test_key_store_and_ledger_share_one_file(tmp_path: Path) -> None:
    """Both open the same SQLite file; WAL keeps them from blocking."""
    db = tmp_path / "gw.db"
    store = KeyStore(db)
    ledger = UsageLedger(db)

    full_key, record = store.create(customer="acme")
    principal = store.authenticate(full_key)
    assert principal is not None
    ledger.record(_event(key_id=principal.key_id, customer=principal.customer))

    assert ledger.summary()[0].key_id == record.key_id
    assert store.get(record.key_id) is not None
    ledger.close()
    store.close()
