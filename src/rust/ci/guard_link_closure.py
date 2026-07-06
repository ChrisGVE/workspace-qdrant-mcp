#!/usr/bin/env python3
"""N14 link-closure guard (default-deny) -- ARCH rev08 §3.3/§9.1.

Read/write separation is only enforceable at crate granularity because Cargo links
whole crate closures. This guard walks the real dependency graph from `cargo
metadata` and fails the build if any client bin's transitive workspace-crate
closure intersects the S1-only set declared in ci/link-policy.toml.

Default-deny: a bin absent from [grants].s1_bins is a client and must stay clear of
every S1-only crate. The S1-only set is intersected with the crates actually
present in the workspace, so on the Phase-0 skeleton (where no S1-only crate exists
yet) the guard passes vacuously and logs that it did -- then bites automatically as
those crates land.

Exit 0 = clean, exit 1 = violation or tooling failure. No silent pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
POLICY = RUST_ROOT / "ci" / "link-policy.toml"


def cargo_metadata() -> dict:
    # Default `cargo metadata` includes the resolved dependency graph (the
    # `resolve` node), which is exactly the closure input this guard needs.
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=RUST_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.stderr.write("guard_link_closure: `cargo metadata` failed:\n")
        sys.stderr.write(out.stderr)
        sys.exit(1)
    return json.loads(out.stdout)


def build_indexes(meta: dict):
    """Return (id->name, id->kinds, name->id, id->dep_ids)."""
    id_name, id_kinds, name_id = {}, {}, {}
    for pkg in meta["packages"]:
        id_name[pkg["id"]] = pkg["name"]
        name_id[pkg["name"]] = pkg["id"]
        id_kinds[pkg["id"]] = {k for t in pkg["targets"] for k in t["kind"]}
    dep_ids = {}
    for node in meta["resolve"]["nodes"]:
        dep_ids[node["id"]] = [d["pkg"] for d in node["deps"]]
    return id_name, id_kinds, name_id, dep_ids


def closure(root_id: str, dep_ids: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        cur = stack.pop()
        for dep in dep_ids.get(cur, ()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def main() -> int:
    policy = tomllib.loads(POLICY.read_text())
    s1_only = set(policy["s1_only"]["crates"])
    s1_bins = set(policy["grants"]["s1_bins"])

    meta = cargo_metadata()
    workspace_ids = set(meta["workspace_members"])
    id_name, id_kinds, name_id, dep_ids = build_indexes(meta)

    workspace_names = {id_name[i] for i in workspace_ids}
    s1_present = s1_only & workspace_names
    if not s1_present:
        print(
            "guard_link_closure: PASS (vacuous) -- no S1-only crate is present in "
            "the workspace yet; nothing for a client bin to illegally link. "
            f"Forward-declared S1-only set: {sorted(s1_only)}."
        )
        return 0

    bins = [i for i in workspace_ids if "bin" in id_kinds.get(i, set())]

    violations: list[str] = []
    checked: list[str] = []
    for bin_id in bins:
        bname = id_name[bin_id]
        if bname in s1_bins:
            continue  # granted the S1 class
        reached = {id_name[i] for i in closure(bin_id, dep_ids) if i in id_name}
        bad = reached & s1_present
        checked.append(bname)
        if bad:
            violations.append(
                f"  client bin `{bname}` links S1-only crate(s): {sorted(bad)}"
            )

    if violations:
        sys.stderr.write(
            "guard_link_closure: FAIL -- the write closure leaked into a client:\n"
            + "\n".join(violations)
            + "\n(ARCH §9.1: a client bin's closure must be disjoint from the "
            "S1-only set. Fix the offending dependency or grant the bin the S1 "
            "class in ci/link-policy.toml only if it is truly a daemon.)\n"
        )
        return 1

    print(
        "guard_link_closure: PASS -- client bins "
        f"{sorted(checked)} are disjoint from present S1-only crates "
        f"{sorted(s1_present)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
