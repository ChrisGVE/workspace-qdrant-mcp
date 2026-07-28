#!/usr/bin/env python3
"""§9.1 whole-workspace crate-graph acyclicity guard (ARCH rev15, R3-N-3).

§9.1 publishes a seven-level stratification of the library crates and then makes a
claim about it: "Every workspace crate's wqm-crate dependencies point strictly DOWN
this stratification, so the crate graph is acyclic and the workspace Cargo-buildable
as specified." It also calls that proof "the standing check" for §8.2's storage seam.

Measured at P04-GT001-WO006: nothing checked it. Cargo rejects an outright cycle at
build time, so the *acyclicity* half was covered by the compiler -- but the
stratification is strictly stronger than acyclicity. An edge that points UP one
level is acyclic and still forbidden: it is exactly the shape round 2 of the
architecture loop found twice (R2-MF-B, the two literal Cargo-rejected cycles), and
the reason the stratification exists rather than a bare "no cycles" rule.

Four checks, each reporting its own line:

  1. Coverage -- every library crate present in the workspace carries a declared
     stratum. A crate nobody stratified is a crate outside the proof, and the proof
     claims to be whole-workspace. Crates on the declared `unstratified` list FAIL
     with the reason (a routed open question), not silently.
  2. Direction -- every wqm-crate dependency of a stratified crate lands on a
     STRICTLY LOWER stratum. Same-stratum edges are violations too: two crates at
     one level with an edge between them are mis-stratified, and same-level edges
     are how a cycle enters.
  3. Acyclicity -- an independent cycle search over the workspace-crate subgraph.
     Redundant with (2) while (2) is clean, and deliberately so: it is the claim
     §9.1 actually makes, and it keeps reporting if the stratum table is ever
     loosened.
  4. Test-support isolation -- no shipped crate may depend on a crate declared
     test-support. Those crates are excused from the stratification because they are
     not part of the shipped graph §9.1 proves acyclic; this check is what keeps that
     premise true, so the exemption is a boundary rather than a hole.

Exit 0 = every check clean, exit 1 = any violation or tooling failure.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from guard_link_closure import Ctx, biting, cargo_metadata

RUST_ROOT = Path(__file__).resolve().parent.parent
POLICY = RUST_ROOT / "ci" / "link-policy.toml"

# The stratification is published as seven levels; the policy names them s0..s6.
STRATUM_KEYS = ("s0", "s1", "s2", "s3", "s4", "s5", "s6")


def test_support(policy: dict) -> set[str]:
    """Crates that are in the workspace but deliberately outside §9.1's map."""
    return set(policy.get("strata", {}).get("test_support", []))


def load_strata(policy: dict) -> tuple[dict[str, int], dict[str, str]]:
    """Return (crate -> stratum, crate -> unstratified reason)."""
    section = policy.get("strata", {})
    strata: dict[str, int] = {}
    for level, key in enumerate(STRATUM_KEYS):
        for crate in section.get(key, []):
            if crate in strata:
                # A crate at two strata makes "strictly down" ambiguous, so the
                # table itself is the defect -- report it rather than pick one.
                sys.stderr.write(
                    f"guard_stratification: policy defect -- `{crate}` is declared "
                    f"at both stratum {strata[crate]} and {level}.\n"
                )
                sys.exit(1)
            strata[crate] = level
    reason = section.get("unstratified_reason", "no reason recorded")
    return strata, {c: reason for c in section.get("unstratified", [])}


def library_names(ctx: Ctx) -> set[str]:
    """Workspace members that are libraries (bins are governed by [grants])."""
    return {
        ctx.id_name[i] for i in ctx.workspace_ids if "lib" in ctx.id_kinds.get(i, set())
    }


def check_coverage(
    ctx: Ctx, strata: dict[str, int], unstratified: dict[str, str], policy: dict
) -> tuple[bool, str]:
    present = library_names(ctx)
    declared = set(strata)
    exempt = test_support(policy)
    blocked = sorted(present & set(unstratified))
    unknown = sorted(present - declared - set(unstratified) - exempt)

    problems = []
    if blocked:
        for crate in blocked:
            problems.append(f"  `{crate}` is UNSTRATIFIED: {unstratified[crate]}")
    if unknown:
        problems.append(
            f"  {unknown} are in the workspace but in no stratum and on no declared "
            "exception list -- either §9.1 stratifies them or a CR says why not."
        )
    if problems:
        return False, (
            "FAIL -- the stratification does not cover the workspace it claims to:\n"
            + "\n".join(problems)
            + "\n(ARCH rev15 §9.1's closing claim is whole-workspace. A crate outside "
            "the table is outside the proof.)"
        )
    exempt_present = sorted(exempt & present)
    exempt_note = (
        f" {exempt_present} are test-support, outside §9.1's shipped-crate map by "
        "declaration -- check 4 enforces that no shipped crate links them."
        if exempt_present
        else ""
    )
    pending = sorted(set(unstratified) - present)
    note = (
        f" {pending} is unstratified pending a CR and will FAIL on arrival "
        f"(expected at {biting(policy, 'unstratified_crate')})."
        if pending
        else ""
    )
    return True, (
        f"PASS -- all {len(present)} library crate(s) present "
        f"({sorted(present)}) carry a declared stratum; "
        f"{len(declared - present)} more are forward-declared.{exempt_note}{note}"
    )


def check_direction(ctx: Ctx, strata: dict[str, int], policy: dict) -> tuple[bool, str]:
    present = library_names(ctx) & set(strata)
    violations, edges = [], 0
    for crate in sorted(present):
        # DIRECT dependencies only: the stratification is an edge-direction rule,
        # and a transitive path that stays strictly-down at every hop is legal.
        crate_id = ctx.name_id[crate]
        for dep_id in ctx.dep_ids.get(crate_id, ()):
            dep = ctx.id_name.get(dep_id)
            if dep is None or dep not in strata or dep == crate:
                continue
            edges += 1
            if strata[dep] >= strata[crate]:
                violations.append(
                    f"  `{crate}` (stratum {strata[crate]}) depends on `{dep}` "
                    f"(stratum {strata[dep]}) -- not strictly down"
                )

    if violations:
        return False, (
            "FAIL -- a wqm-crate dependency does not point strictly down:\n"
            + "\n".join(violations)
            + "\n(ARCH rev15 §9.1: every workspace crate's wqm-crate dependencies "
            "point strictly DOWN the stratification. An up-edge or a same-level edge "
            "is a mis-stratification or the start of a cycle -- both are CRs to P02, "
            "not policy edits.)"
        )
    if edges == 0:
        return True, (
            f"PASS (vacuous) -- {len(present)} stratified crate(s) present "
            f"({sorted(present)}) but no wqm-crate edge exists between them yet. "
            "Bites at the first inter-crate dependency, expected at "
            f"{biting(policy, 'stratification_edges')} (where the strata start filling)."
        )
    return True, (
        f"PASS -- {edges} wqm-crate edge(s) across {len(present)} stratified crate(s) "
        "all point strictly down."
    )


def check_acyclic(ctx: Ctx) -> tuple[bool, str]:
    """Independent cycle search over the workspace-crate subgraph."""
    names = library_names(ctx) | {
        ctx.id_name[i] for i in ctx.workspace_ids if i in ctx.id_name
    }
    adjacency = {
        n: sorted(
            ctx.id_name[d]
            for d in ctx.dep_ids.get(ctx.name_id[n], ())
            if ctx.id_name.get(d) in names and ctx.id_name[d] != n
        )
        for n in sorted(names)
    }

    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(adjacency, WHITE)
    cycle: list[str] = []

    def visit(node: str, path: list[str]) -> bool:
        color[node] = GREY
        for nxt in adjacency[node]:
            if color[nxt] == GREY:
                cycle.extend(path[path.index(nxt) :] + [nxt])
                return True
            if color[nxt] == WHITE and visit(nxt, path + [nxt]):
                return True
        color[node] = BLACK
        return False

    for node in adjacency:
        if color[node] == WHITE and visit(node, [node]):
            return False, (
                "FAIL -- the workspace crate graph contains a cycle: "
                + " -> ".join(cycle)
                + "\n(Cargo would reject this too; §9.1's stratification exists so it "
                "is caught as a design violation with a named direction, not as an "
                "opaque build error.)"
            )
    return True, (
        f"PASS -- the {len(adjacency)}-crate workspace subgraph is acyclic "
        f"({sum(len(v) for v in adjacency.values())} edge(s) walked)."
    )


def check_test_support_isolation(
    ctx: Ctx, strata: dict[str, int], policy: dict
) -> tuple[bool, str]:
    """4. No shipped crate may depend on a test-support crate.

    This is what makes the [strata].test_support exemption a boundary rather than a
    hole. A test harness is excused from the acyclicity proof because it is not part
    of the shipped graph -- the moment a shipped crate links it, that premise is
    false and the exemption has to be re-argued, not silently extended.
    """
    exempt = test_support(policy)
    present = library_names(ctx) & set(strata)
    exempt_present = exempt & ctx.workspace_names

    if not exempt_present:
        return True, (
            f"PASS (vacuous) -- no test-support crate is present (declared: "
            f"{sorted(exempt) or 'none'}). Bites when one lands."
        )

    violations = []
    for crate in sorted(present):
        bad = ctx.wqm_closure(crate) & exempt_present
        if bad:
            violations.append(f"  shipped crate `{crate}` links test-support {sorted(bad)}")

    if violations:
        return False, (
            "FAIL -- a shipped crate depends on test-support code:\n"
            + "\n".join(violations)
            + "\n(A test-support crate is exempt from §9.1's stratification only "
            "because nothing shipped links it. Move the shared code into a stratified "
            "crate, or bring the harness into the map via a CR to P02.)"
        )
    return True, (
        f"PASS -- no shipped crate among {sorted(present)} links test-support "
        f"{sorted(exempt_present)}."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§9.1 stratification / acyclicity guard")
    ap.add_argument("--workspace", type=Path, default=RUST_ROOT)
    ap.add_argument("--policy", type=Path, default=POLICY)
    args = ap.parse_args(argv)

    policy = tomllib.loads(args.policy.read_text())
    strata, unstratified = load_strata(policy)
    ctx = Ctx(cargo_metadata(args.workspace))

    results = (
        ("1 coverage", check_coverage(ctx, strata, unstratified, policy)),
        ("2 strictly-down direction", check_direction(ctx, strata, policy)),
        ("3 acyclicity", check_acyclic(ctx)),
        (
            "4 test-support isolation",
            check_test_support_isolation(ctx, strata, policy),
        ),
    )

    failed = False
    for label, (ok, message) in results:
        stream = sys.stdout if ok else sys.stderr
        stream.write(f"guard_stratification [{label}]: {message}\n")
        if not ok:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
