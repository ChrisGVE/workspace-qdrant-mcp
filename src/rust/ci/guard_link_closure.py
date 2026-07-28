#!/usr/bin/env python3
"""N14 link-closure guard (default-deny) -- ARCH rev15 §3.3/§9.1, N14 row §7.

Read/write separation is only enforceable at crate granularity because Cargo links
whole crate closures. This guard walks the real dependency graph from `cargo
metadata` and enforces the closure rules the architecture declares machine-checked.

N14's guard walks the closure in BOTH directions (rev11, AGP-11). Three
independent checks, each reporting its own PASS / PASS (vacuous) / FAIL:

  A. Client direction -- no client bin's transitive workspace-crate closure may
     intersect the S1-only set. Default-deny: a bin absent from every [grants]
     list is a client. The restore maintenance binary is the one declared
     exception (ARCH rev15 §3.4): it is NOT default-denied, and its closure is
     checked against its own declared allow-list instead.
  B. Kernel purity -- no KERNEL crate's closure may contain `wqm-conventions`.
     The kernel reads product-convention data through `wqm-common` traits,
     N28-injected; a link edge to the conventions crate is the amorphous-
     collections seam reopening.
  C. Sensor agnosticism -- `wqm-sensor`'s closure may contain only `wqm-common`
     among wqm crates. Product-agnosticism as a link fact (§3.4 boundary card).
     External crates (notify, notify-debouncer-full, the branch-mgmt lib) are not
     constrained.
  D. The storage/algorithm seam (§8.2, DP-7) -- no storage crate's closure may
     contain an algorithm/engine crate, and `wqm-store-write`'s closure is exactly
     the four crates §9.1 enumerates. §8.2 asserted "the N14 guard covers this
     direction too"; measured at P04-GT001-WO006, it did not. This check is that
     sentence made true.

Each check is vacuous only while the crates it governs are absent, and says so --
a green run never hides an inert check. Exit 0 = every check clean, exit 1 = any
violation or tooling failure. No silent pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
POLICY = RUST_ROOT / "ci" / "link-policy.toml"


def cargo_metadata(root: Path) -> dict:
    # Default `cargo metadata` includes the resolved dependency graph (the
    # `resolve` node), which is exactly the closure input this guard needs.
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.stderr.write("guard_link_closure: `cargo metadata` failed:\n")
        sys.stderr.write(out.stderr)
        sys.exit(1)
    return json.loads(out.stdout)


def is_shipped_edge(dep: dict) -> bool:
    """Whether a resolved dependency edge is part of the SHIPPED graph.

    Cargo classifies each edge as normal, `dev` or `build` (`dep_kinds[].kind`,
    where a normal edge carries `null`). Only normal and build edges end up in what
    `cargo build` produces; a dev-dependency is compiled solely for `cargo test`.

    The distinction is load-bearing for every check in this file, and it was
    MISSING until `P04-GT001-WO013`. Each rule here is about what a shipped
    artifact links -- an S1-only crate inside a client binary, the conventions
    crate inside the kernel, a test harness inside a product crate. A dev-only edge
    is none of those, and Cargo agrees so completely that it permits dev-dependency
    CYCLES, which the acyclicity rule this guard's sibling enforces would otherwise
    have to forbid.

    The evidence that the omission was an oversight rather than a policy is that
    the guards already disagreed with each other about it: `guard_stratification`
    check 4 fired on a library crate's dev-dependency on the test harness while the
    identical relationship on a *bin* was invisible, because bins are outside that
    check's subject. One relationship, two verdicts, decided by which target
    happened to declare it. Under the shipped-graph reading both are legal, and
    they are legal for the same reason.

    An edge with no `dep_kinds` (older metadata) is treated as shipped: unknown
    provenance defaults to the stricter answer.
    """
    kinds = dep.get("dep_kinds")
    if not kinds:
        return True
    return any(k.get("kind") in (None, "build") for k in kinds)


def build_indexes(meta: dict, *, include_dev: bool = False):
    """Return (id->name, id->kinds, name->id, id->dep_ids).

    `include_dev=True` walks every edge, which is what a check about the *test*
    graph would want. No check wants it today; the flag exists so that asking for
    the other graph is a choice with a name rather than a second index builder.
    """
    id_name, id_kinds, name_id = {}, {}, {}
    for pkg in meta["packages"]:
        id_name[pkg["id"]] = pkg["name"]
        name_id[pkg["name"]] = pkg["id"]
        id_kinds[pkg["id"]] = {k for t in pkg["targets"] for k in t["kind"]}
    dep_ids = {}
    for node in meta["resolve"]["nodes"]:
        dep_ids[node["id"]] = [
            d["pkg"] for d in node["deps"] if include_dev or is_shipped_edge(d)
        ]
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


class Ctx:
    """The resolved graph plus the name lookups every check shares."""

    def __init__(self, meta: dict):
        self.workspace_ids = set(meta["workspace_members"])
        (self.id_name, self.id_kinds, self.name_id, self.dep_ids) = build_indexes(meta)
        self.workspace_names = {self.id_name[i] for i in self.workspace_ids}

    def wqm_closure(self, crate: str) -> set[str]:
        """Workspace-crate names reachable from `crate`, excluding itself."""
        root = self.name_id[crate]
        reached = {
            self.id_name[i] for i in closure(root, self.dep_ids) if i in self.id_name
        }
        return (reached & self.workspace_names) - {crate}


def biting(policy: dict, key: str) -> str:
    """The engagement at which a vacuous check activates (`[biting]` in the policy).

    Charter §2.2: a vacuous check must name where it starts biting, not only the
    condition. Absent from the policy, say so rather than inventing an engagement --
    an unknown answer is information, a guessed one is not.
    """
    return policy.get("biting", {}).get(key, "an engagement the policy does not record")


def check_clients(ctx: Ctx, policy: dict) -> tuple[bool, str]:
    """A. No client bin's closure may intersect the S1-only set."""
    s1_only = set(policy["s1_only"]["crates"])
    grants = policy["grants"]
    s1_bins = set(grants["s1_bins"])
    restore_bins = set(grants.get("restore_bins", []))
    restore_allowed = set(grants.get("restore_allowed", []))

    s1_present = s1_only & ctx.workspace_names
    if not s1_present:
        return True, (
            "PASS (vacuous) -- no S1-only crate is present in the workspace yet; "
            "nothing for a client bin to illegally link. Bites when the first of "
            f"{sorted(s1_only)} lands, expected at {biting(policy, 'client_direction')}. "
            "Forward-declared by P04-GT001-WO002."
        )

    bins = [i for i in ctx.workspace_ids if "bin" in ctx.id_kinds.get(i, set())]
    violations, checked = [], []
    for bin_id in bins:
        bname = ctx.id_name[bin_id]
        if bname in s1_bins:
            continue  # granted the S1 class
        reached = ctx.wqm_closure(bname)
        if bname in restore_bins:
            # Declared, not default-denied: checked against its OWN allow-list.
            bad = reached - restore_allowed
            checked.append(f"{bname}(restore)")
            if bad:
                violations.append(
                    f"  restore bin `{bname}` links crate(s) outside its declared "
                    f"closure: {sorted(bad)}"
                )
            continue
        bad = reached & s1_present
        checked.append(bname)
        if bad:
            violations.append(
                f"  client bin `{bname}` links S1-only crate(s): {sorted(bad)}"
            )

    if violations:
        return False, (
            "FAIL -- the write closure leaked into a client:\n"
            + "\n".join(violations)
            + "\n(ARCH rev15 §9.1: a client bin's closure must be disjoint from the "
            "S1-only set. Fix the offending dependency, or grant the bin the S1 "
            "class in ci/link-policy.toml only if it is truly a daemon.)"
        )
    return True, (
        f"PASS -- non-S1 bins {sorted(checked)} respect their closures against "
        f"present S1-only crates {sorted(s1_present)}."
    )


def check_kernel_purity(ctx: Ctx, policy: dict) -> tuple[bool, str]:
    """B. No kernel crate's closure may contain a forbidden crate (AGP-11)."""
    section = policy.get("kernel_purity", {})
    kernel = set(section.get("kernel_crates", []))
    forbidden = set(section.get("forbidden_in_kernel", []))

    kernel_present = kernel & ctx.workspace_names
    forbidden_present = forbidden & ctx.workspace_names
    if not kernel_present or not forbidden_present:
        return True, (
            "PASS (vacuous) -- the kernel/conventions pair is not both present yet "
            f"(kernel crates present: {sorted(kernel_present) or 'none'}; forbidden "
            f"present: {sorted(forbidden_present) or 'none'}). Bites when a kernel "
            f"crate and one of {sorted(forbidden)} coexist, expected at "
            f"{biting(policy, 'kernel_purity')}. Added by P04-GT001-WO005 "
            "(AGP-11 second direction)."
        )

    violations = []
    for crate in sorted(kernel_present):
        bad = ctx.wqm_closure(crate) & forbidden_present
        if bad:
            violations.append(f"  kernel crate `{crate}` links {sorted(bad)}")

    if violations:
        return False, (
            "FAIL -- a kernel crate reaches product-convention data by link:\n"
            + "\n".join(violations)
            + "\n(ARCH rev15 AGP-11: the kernel reads convention data through "
            "`wqm-common` traits, N28-injected -- never by depending on "
            "`wqm-conventions`.)"
        )
    return True, (
        f"PASS -- kernel crates {sorted(kernel_present)} are free of "
        f"{sorted(forbidden_present)}."
    )


def check_sensor(ctx: Ctx, policy: dict) -> tuple[bool, str]:
    """C. wqm-sensor's wqm-crate closure may contain only its allow-list."""
    section = policy.get("kernel_purity", {}).get("sensor", {})
    crate = section.get("crate")
    allowed = set(section.get("allowed_wqm_deps", []))

    if not crate or crate not in ctx.workspace_names:
        return True, (
            f"PASS (vacuous) -- `{crate or 'wqm-sensor'}` is not in the workspace "
            f"yet. Bites when it lands, expected at "
            f"{biting(policy, 'sensor_agnosticism')}. Added by P04-GT001-WO005."
        )

    bad = ctx.wqm_closure(crate) - allowed
    if bad:
        return False, (
            f"FAIL -- `{crate}` links wqm crate(s) outside its declared closure: "
            f"{sorted(bad)}\n(ARCH rev15 §3.4 boundary card: the sensor is "
            f"product-agnostic as a LINK FACT; its wqm closure is exactly "
            f"{sorted(allowed)} plus external crates.)"
        )
    return True, f"PASS -- `{crate}`'s wqm closure is within {sorted(allowed)}."


def check_storage_algorithm(ctx: Ctx, policy: dict) -> tuple[bool, str]:
    """D. No storage crate may reach an algorithm crate (§8.2, DP-7).

    Two assertions from the same §8.2/§9.1 storage-seam paragraph: the general
    direction rule, and `wqm-store-write`'s exactly-enumerated closure. The exact
    form has teeth the disjointness form lacks -- it also rejects a storage edge to
    a crate that is neither an algorithm crate nor in the enumeration.
    """
    section = policy.get("dp7", {})
    storage = set(section.get("storage_crates", []))
    algorithms = set(section.get("algorithm_crates", []))
    exact = section.get("exact_closures", {})

    storage_present = storage & ctx.workspace_names
    if not storage_present:
        return True, (
            "PASS (vacuous) -- no storage crate is present in the workspace yet "
            f"(awaiting {sorted(storage)}). Bites when the first one lands, expected "
            f"at {biting(policy, 'storage_algorithm')}. Added by P04-GT001-WO006: "
            "§8.2 claimed the N14 guard already walked this direction; it did not."
        )

    violations, checked = [], []
    for crate in sorted(storage_present):
        reached = ctx.wqm_closure(crate)
        checked.append(crate)
        bad = reached & algorithms
        if bad:
            violations.append(
                f"  storage crate `{crate}` links algorithm crate(s): {sorted(bad)}"
            )
        declared = exact.get(crate)
        if declared is not None:
            surplus = reached - set(declared)
            if surplus:
                violations.append(
                    f"  storage crate `{crate}` exceeds its §9.1 exact closure "
                    f"{sorted(declared)} by {sorted(surplus)}"
                )

    if violations:
        return False, (
            "FAIL -- the storage/algorithm seam is crossed the wrong way:\n"
            + "\n".join(violations)
            + "\n(ARCH rev15 §8.2/DP-7: an algorithm crate may depend on its "
            "storage owner's contract; nothing in the storage owner may depend on "
            "an algorithm crate. Cross-seam invocations are injected ports whose "
            "traits live in `wqm-common` -- never links.)"
        )
    exact_note = (
        f" Exact closures enforced for {sorted(set(exact) & storage_present)}."
        if set(exact) & storage_present
        else ""
    )
    return True, (
        f"PASS -- storage crates {sorted(checked)} reach no algorithm crate "
        f"{sorted(algorithms)}.{exact_note}"
    )


CHECKS = (
    ("A client-direction (S1-only disjointness)", check_clients),
    ("B kernel purity (AGP-11)", check_kernel_purity),
    ("C sensor agnosticism", check_sensor),
    ("D storage/algorithm direction (§8.2, DP-7)", check_storage_algorithm),
)


def main(argv: list[str] | None = None) -> int:
    # The overrides exist so `selftest_link_closure.py` can point the guard at a
    # fixture workspace carrying a deliberate violation and prove each check
    # FAILS where it claims to. Without that, three vacuous PASS lines assert
    # teeth nobody has ever seen bite.
    ap = argparse.ArgumentParser(description="N14 link-closure guard")
    ap.add_argument(
        "--workspace",
        type=Path,
        default=RUST_ROOT,
        help="cargo workspace root to inspect (default: src/rust)",
    )
    ap.add_argument(
        "--policy",
        type=Path,
        default=POLICY,
        help="link-policy.toml to enforce (default: ci/link-policy.toml)",
    )
    args = ap.parse_args(argv)

    policy = tomllib.loads(args.policy.read_text())
    ctx = Ctx(cargo_metadata(args.workspace))

    failed = False
    for label, fn in CHECKS:
        ok, message = fn(ctx, policy)
        stream = sys.stdout if ok else sys.stderr
        stream.write(f"guard_link_closure [{label}]: {message}\n")
        if not ok:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
