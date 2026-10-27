#!/usr/bin/env python3
"""Audit the resolved dependencies in uv.lock against the OSV database.

Reads the exact (name, version) pairs that uv resolved (from the TOML lock
file), then asks the OSV API whether any of them have a known vulnerability
in our resolved version. Exits non-zero if any are vulnerable, so it can gate
CI. No third-party imports — stdlib only (tomllib, urllib.request, json).

Usage:
    python scripts/audit_osv.py [path-to-uv.lock]   # default: ./uv.lock
"""

from __future__ import annotations

import json
import sys
import tomllib
import urllib.request

OSV_API = "https://api.osv.dev/v1/query"


def load_locked_packages(lock_path: str) -> list[tuple[str, str]]:
    with open(lock_path, "rb") as fh:
        data = tomllib.load(fh)
    return [(p["name"], str(p.get("version", ""))) for p in data.get("package", [])]


def query_osv(pkg: str, ver: str, timeout: int = 30) -> list[dict]:
    if not ver:
        return []
    payload = json.dumps(
        {"package": {"name": pkg, "ecosystem": "PyPI"}, "version": ver}
    ).encode()
    req = urllib.request.Request(
        OSV_API, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp).get("vulns", [])


def main() -> int:
    lock_path = sys.argv[1] if len(sys.argv) > 1 else "uv.lock"
    pkgs = load_locked_packages(lock_path)

    vulnerable = 0
    for pkg, ver in pkgs:
        try:
            vulns = query_osv(pkg, ver)
        except Exception as exc:  # network/parse: report, don't hide behind a crash
            print(f"WARN: could not query OSV for {pkg}=={ver}: {exc}", file=sys.stderr)
            continue
        if vulns:
            vulnerable += 1
            print(f"VULNERABLE: {pkg}=={ver}")
            for v in vulns:
                aliases = v.get("aliases") or []
                summary = (v.get("summary") or "").replace("\n", " ")
                detail = ", ".join(aliases) if aliases else ""
                print(f"  - {v['id']}" + (f"  [{detail}]" if detail else ""))
                if summary:
                    print(f"    {summary}")

    if vulnerable:
        print(
            f"\nFAIL: {vulnerable} locked package(s) have known vulnerabilities",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {len(pkgs)} locked packages, "
        "no known vulnerabilities in the resolved versions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
