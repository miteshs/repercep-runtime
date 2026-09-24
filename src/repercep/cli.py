"""The ``repercep`` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_USAGE = "usage: repercep [info|keys|usage] ..."


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "info"

    if command in ("-h", "--help", "help"):
        print(_USAGE)
        print("\n  info    print the detected backend and devices")
        print("  keys    manage per-customer API keys (create/list/revoke)")
        print("  usage   report metered usage from the ledger")
        return 0
    if command == "info":
        return _info()
    if command == "keys":
        return _keys(args[1:])
    if command == "usage":
        return _usage(args[1:])

    print(f"repercep: unknown command {command!r}", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


def _info() -> int:
    """Print the detected backend and devices."""
    from repercep import __version__
    from repercep.backend.registry import select_backend

    print(f"Repercep Runtime v{__version__}")
    try:
        backend = select_backend()
    except RuntimeError as exc:
        print("  backend : NONE")
        print(f"  reason  : {exc}")
        return 1

    caps = backend.capabilities()
    print(f"  backend : {backend.name} ({backend.vendor.value})")
    for dev in backend.devices():
        print(
            f"  device  : [{dev.index}] {dev.name}  {dev.arch}  "
            f"{dev.total_memory_gib:.0f} GiB  {dev.multi_processor_count} CUs"
        )
    print(f"  dtype   : {backend.default_dtype().value}")
    print(f"  attn    : {', '.join(caps.attention_ops)}")
    print(f"  fp8     : {caps.supports_fp8}   flash-attn: {caps.supports_flash_attention}")
    return 0


def _db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the gateway SQLite file (REPERCEP_GATEWAY_DB).",
    )


def _keys(argv: Sequence[str]) -> int:
    """Manage per-customer API keys.

    Key minting is CLI-only on purpose: there is no HTTP endpoint that can
    create or list credentials, so a stolen customer key cannot be escalated
    into minting more.
    """
    from repercep.serving.tenancy import KeyStore

    parser = argparse.ArgumentParser(prog="repercep keys")
    sub = parser.add_subparsers(dest="action", required=True)

    create = sub.add_parser("create", help="mint a key for a customer")
    _db_argument(create)
    create.add_argument("--customer", required=True, help="billing entity for this key")
    create.add_argument("--label", default="", help="free-text note, e.g. 'staging'")
    create.add_argument(
        "--monthly-token-quota",
        type=int,
        default=None,
        help="reject calls once this many tokens are billed in a UTC month",
    )

    listing = sub.add_parser("list", help="list keys (never shows secrets)")
    _db_argument(listing)
    listing.add_argument("--customer", default=None, help="filter to one customer")
    listing.add_argument("--active", action="store_true", help="hide revoked keys")

    revoke = sub.add_parser("revoke", help="revoke a key by id")
    _db_argument(revoke)
    revoke.add_argument("key_id")

    args = parser.parse_args(list(argv))
    store = KeyStore(args.db)
    try:
        if args.action == "create":
            full_key, record = store.create(
                customer=args.customer,
                label=args.label,
                monthly_token_quota=args.monthly_token_quota,
            )
            print(f"key_id   : {record.key_id}")
            print(f"customer : {record.customer}")
            if record.label:
                print(f"label    : {record.label}")
            if record.monthly_token_quota is not None:
                print(f"quota    : {record.monthly_token_quota} tokens/month")
            print(f"\n  {full_key}\n")
            print("Store it now — only its hash is kept, so it cannot be shown again.")
            return 0

        if args.action == "list":
            rows = store.list_keys(customer=args.customer, include_revoked=not args.active)
            if not rows:
                print("no keys")
                return 0
            print(f"{'KEY ID':<10} {'CUSTOMER':<20} {'STATUS':<8} {'QUOTA':>12}  LABEL")
            for r in rows:
                quota = "-" if r.monthly_token_quota is None else f"{r.monthly_token_quota:,}"
                status = "active" if r.active else "revoked"
                print(f"{r.key_id:<10} {r.customer:<20} {status:<8} {quota:>12}  {r.label}")
            return 0

        if store.revoke(args.key_id):
            print(f"revoked {args.key_id}")
            return 0
        print(f"repercep: no active key {args.key_id!r}", file=sys.stderr)
        return 1
    finally:
        store.close()


def _usage(argv: Sequence[str]) -> int:
    """Report metered usage — the invoice query."""
    from repercep.serving.usage import UsageLedger, month_start

    parser = argparse.ArgumentParser(prog="repercep usage")
    _db_argument(parser)
    parser.add_argument("--customer", default=None, help="filter to one customer")
    parser.add_argument(
        "--all-time",
        action="store_true",
        help="report over all history instead of the current UTC month",
    )
    args = parser.parse_args(list(argv))

    ledger = UsageLedger(args.db)
    try:
        since = None if args.all_time else month_start()
        rows = ledger.summary(since=since, customer=args.customer)
        print(f"period   : {'all time' if since is None else since + ' →'}")
        if not rows:
            print("no usage recorded")
            return 0
        print(
            f"\n{'KEY ID':<10} {'CUSTOMER':<20} {'CALLS':>8} "
            f"{'PROMPT':>12} {'COMPLETION':>12} {'TOTAL':>12} {'EST':>10}"
        )
        totals = [0, 0, 0, 0]
        for r in rows:
            print(
                f"{r.key_id:<10} {r.customer:<20} {r.calls:>8,} "
                f"{r.prompt_tokens:>12,} {r.completion_tokens:>12,} "
                f"{r.total_tokens:>12,} {r.estimated_tokens:>10,}"
            )
            totals[0] += r.calls
            totals[1] += r.prompt_tokens
            totals[2] += r.completion_tokens
            totals[3] += r.estimated_tokens
        print(
            f"{'TOTAL':<10} {'':<20} {totals[0]:>8,} "
            f"{totals[1]:>12,} {totals[2]:>12,} {totals[1] + totals[2]:>12,} {totals[3]:>10,}"
        )
        if totals[3]:
            print(
                f"\nNote: {totals[3]:,} tokens are estimated, not engine-reported "
                "(streaming calls where no usage was available). Disclose this on any "
                "invoice derived from it — see docs/METERED_GATEWAY.md."
            )
        return 0
    finally:
        ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
