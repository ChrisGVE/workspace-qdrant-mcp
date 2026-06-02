#!/usr/bin/env python3
"""Admin helpers for the Linux/WSL Makefile (container-first flow).

Usage:
  python3 scripts/wqm_admin.py tenants      [BASE_URL]            # one tenantId per line
  python3 scripts/wqm_admin.py status       [BASE_URL]            # per-project indexing rows
  python3 scripts/wqm_admin.py scan         [BASE_URL]            # discovered git repos under devRoot
  python3 scripts/wqm_admin.py register-all [BASE_URL]            # register every scanned candidate
  python3 scripts/wqm_admin.py watch        [BASE_URL] [INT] [MAX]  # poll until indexing drains

`watch` polls every INT seconds (default 10) up to MAX seconds (default 900),
printing a timestamped table each tick and exiting early when every project is
at 100% with 0 pending / 0 in-progress.

BASE_URL defaults to http://localhost:6335. Host-side calls rely on
MCP_HTTP_TRUST_LOCALHOST=1 (the compose default), which trusts loopback peers,
so no Bearer token is sent. Uses only the Python standard library (no jq).
"""
import json
import sys
import time
import urllib.request

DEFAULT_BASE = "http://localhost:6335"


def _get(base, path, timeout=10):
    url = base.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post(base, path, payload, timeout=60):
    url = base.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch_projects(base):
    """Return the flat list of project objects from /admin/api/snapshot.

    The snapshot's `projects` field is a dict mixing list buckets (e.g.
    `registered`) with scalar summaries (e.g. `registeredCount`). Flatten only
    the list-valued buckets into a single list of project objects.
    """
    data = _get(base, "/admin/api/snapshot")
    buckets = data.get("projects", {}).values()
    return [p for bucket in buckets if isinstance(bucket, list) for p in bucket]


def scan_candidates(base):
    """POST /admin/api/projects/scan and return the list of candidate dicts."""
    data = _post(base, "/admin/api/projects/scan", {}, timeout=30)
    return data.get("scan", {}).get("candidates", [])


def _int(v, default=0):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def format_row(p):
    ix = p.get("indexing", {})
    return (
        "{tenant}  {pct:>3}%  done={done}/{total}  pending={pending}  "
        "failed={failed}  {path}".format(
            tenant=p.get("tenantId", "?"),
            pct=_int(ix.get("percent", 0)),
            done=ix.get("done", "?"),
            total=ix.get("total", "?"),
            pending=ix.get("pending", "?"),
            failed=ix.get("failed", "?"),
            path=p.get("path", "?"),
        )
    )


def project_done(p):
    """True when a project has nothing pending or in-progress and is at 100%."""
    ix = p.get("indexing", {})
    return (
        _int(ix.get("pending", 0)) == 0
        and _int(ix.get("in_progress", 0)) == 0
        and _int(ix.get("percent", 0)) >= 100
        and _int(ix.get("total", 0)) > 0
    )


def cmd_status(base):
    rows = fetch_projects(base)
    if not rows:
        print("no watched projects")
        return
    for p in rows:
        print(format_row(p))


def cmd_watch(base, interval, max_seconds):
    deadline = time.monotonic() + max_seconds
    tick = 0
    while True:
        tick += 1
        rows = fetch_projects(base)
        stamp = time.strftime("%H:%M:%S")
        total_pending = sum(_int(p.get("indexing", {}).get("pending", 0)) for p in rows)
        total_failed = sum(_int(p.get("indexing", {}).get("failed", 0)) for p in rows)
        print("--- tick {} {}  (projects={} pending={} failed={}) ---".format(
            tick, stamp, len(rows), total_pending, total_failed))
        for p in rows:
            print("  " + format_row(p))
        if rows and all(project_done(p) for p in rows):
            print("\nAll {} projects fully indexed (0 pending, 100%).".format(len(rows)))
            return 0
        if time.monotonic() >= deadline:
            # Timeout is not a failure for a monitoring command — exit 0 so
            # `make watch` doesn't surface a scary non-zero status. Re-run to
            # keep watching.
            print("\nwatch window of {}s elapsed — {} item(s) still pending. "
                  "Re-run 'make watch' to keep following.".format(max_seconds, total_pending))
            return 0
        time.sleep(interval)


def cmd_register_all(base):
    candidates = scan_candidates(base)
    if not candidates:
        print("no git repos discovered under the configured devRoot")
        return
    ok = 0
    for c in candidates:
        path = c.get("path")
        name = c.get("name", path)
        try:
            resp = _post(
                base, "/admin/api/projects/register",
                {"path": path, "registerIfNew": True},
            )
            tenant = resp.get("tenantId") or resp.get("tenant_id") or "?"
            status = "ok" if resp.get("ok", True) else "rejected"
            print("[{}] {:<22} tenant={} {}".format(status, name, tenant, path))
            ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print("[fail] {:<22} {} -> {}".format(name, path, exc))
    print("\nRegistered/refreshed {}/{} candidates.".format(ok, len(candidates)))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    base = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASE

    if cmd == "tenants":
        for p in fetch_projects(base):
            print(p["tenantId"])
    elif cmd == "scan":
        for c in scan_candidates(base):
            print("{:<22} {:<10} {}".format(c.get("name", "?"), c.get("branch", "?"), c.get("path", "?")))
    elif cmd == "register-all":
        cmd_register_all(base)
    elif cmd == "watch":
        interval = _int(sys.argv[3], 10) if len(sys.argv) > 3 else 10
        max_seconds = _int(sys.argv[4], 900) if len(sys.argv) > 4 else 900
        sys.exit(cmd_watch(base, interval, max_seconds))
    else:  # status
        cmd_status(base)


if __name__ == "__main__":
    main()
