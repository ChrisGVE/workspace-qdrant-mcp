#!/usr/bin/env python3
"""Prove the N14 link-closure guard FAILS where it claims to.

`guard_link_closure.py` reports three checks, and on today's workspace all three
are vacuous -- no S1-only crate exists yet, no kernel crate, no sensor. Three
vacuous PASS lines assert teeth nobody has ever seen bite, and a guard that has
never failed is indistinguishable from a guard that cannot fail. That is the
false-coverage class this program exists to cure (DELIVERABLES: "false-coverage
tests -> real coverage").

So: build throwaway cargo workspaces carrying one deliberate violation each,
point the guard at them, and assert it exits 1 with the right diagnosis. Then
build a clean one and assert it exits 0 -- because a guard that fails on
everything is no better than one that fails on nothing.

Fixtures are tiny (empty `lib.rs`, no external deps) so `cargo metadata` resolves
offline in milliseconds. Nothing here touches the real workspace.

Exit 0 = every case behaved as claimed, exit 1 = a check did not bite.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
GUARD = CI_DIR / "guard_link_closure.py"

# A policy mirroring the real one's shape, trimmed to the crates the fixtures use.
POLICY = """
[s1_only]
crates = ["wqm-store-write", "wqm-serve", "wqm-lsp", "wqm-sensor"]

[grants]
s1_bins = ["memexd"]
restore_bins = ["wqm-restore"]
restore_allowed = ["wqm-store-write", "wqm-common"]

[kernel_purity]
kernel_crates = ["wqm-store", "wqm-store-write"]
forbidden_in_kernel = ["wqm-conventions"]

[kernel_purity.sensor]
crate = "wqm-sensor"
allowed_wqm_deps = ["wqm-common"]

[dp7]
storage_crates = ["wqm-store", "wqm-store-write"]
algorithm_crates = ["wqm-graph", "wqm-intel"]

[dp7.exact_closures]
wqm-store-write = ["wqm-search", "wqm-store", "wqm-common"]
"""


def write_crate(root: Path, name: str, deps: list[str], is_bin: bool) -> None:
    """Emit a minimal crate with the given workspace-path dependencies.

    A dependency spelled `dev:<name>` becomes a **dev-dependency**, which is how a
    case expresses an edge that exists for `cargo test` and not for `cargo build`.
    The guards read the shipped graph only (`is_shipped_edge`), so the two kinds
    have to be expressible here or that distinction has no test.
    """
    sub = "bins" if is_bin else "crates"
    d = root / sub / name
    (d / "src").mkdir(parents=True)

    def line(dep: str) -> str:
        return '%s = { path = "../../crates/%s" }' % (dep, dep)

    normal = [line(dep) for dep in deps if not dep.startswith("dev:")]
    dev = [line(dep.removeprefix("dev:")) for dep in deps if dep.startswith("dev:")]

    (d / "Cargo.toml").write_text(
        "[package]\n"
        f'name = "{name}"\n'
        'version = "0.0.0"\n'
        'edition = "2021"\n\n'
        "[dependencies]\n" + "\n".join(normal) + "\n\n"
        "[dev-dependencies]\n" + "\n".join(dev) + "\n"
    )
    (d / "src" / ("main.rs" if is_bin else "lib.rs")).write_text(
        "fn main() {}\n" if is_bin else ""
    )


def build_workspace(
    root: Path, libs: dict[str, list[str]], bins: dict[str, list[str]]
) -> None:
    # `members` is built from what actually exists: cargo rejects a `bins/*` glob
    # whose directory is absent, and several cases have no bin at all.
    members = ['"crates/*"'] + (['"bins/*"'] if bins else [])
    (root / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = [%s]\n' % ", ".join(members)
    )
    for name, deps in libs.items():
        write_crate(root, name, deps, is_bin=False)
    for name, deps in bins.items():
        write_crate(root, name, deps, is_bin=True)
    (root / "policy.toml").write_text(POLICY)


def run_guard(root: Path) -> tuple[int, str]:
    out = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--workspace",
            str(root),
            "--policy",
            str(root / "policy.toml"),
        ],
        capture_output=True,
        text=True,
    )
    return out.returncode, out.stdout + out.stderr


# (case name, libs, bins, expect_fail, a substring the diagnosis must contain)
CASES = [
    (
        "A: a client bin links an S1-only crate",
        {"wqm-common": [], "wqm-store-write": ["wqm-common"]},
        {"workspace-qdrant-mcp": ["wqm-store-write"]},
        True,
        "links S1-only crate(s)",
    ),
    (
        "A: the S1-granted daemon may link the same crate",
        {"wqm-common": [], "wqm-store-write": ["wqm-common"]},
        {"memexd": ["wqm-store-write"]},
        False,
        "PASS",
    ),
    (
        "A: the restore bin is declared, not default-denied",
        {"wqm-common": [], "wqm-store-write": ["wqm-common"]},
        {"wqm-restore": ["wqm-store-write"]},
        False,
        "PASS",
    ),
    (
        "A: but the restore bin may not exceed its declared closure",
        {
            "wqm-common": [],
            "wqm-store-write": ["wqm-common"],
            "wqm-serve": ["wqm-common"],
        },
        {"wqm-restore": ["wqm-store-write", "wqm-serve"]},
        True,
        "outside its declared closure",
    ),
    (
        "B: a kernel crate links wqm-conventions (AGP-11)",
        {
            "wqm-common": [],
            "wqm-conventions": ["wqm-common"],
            "wqm-store": ["wqm-conventions"],
        },
        {},
        True,
        "links ['wqm-conventions']",
    ),
    (
        "B: a kernel crate free of wqm-conventions passes",
        {
            "wqm-common": [],
            "wqm-conventions": ["wqm-common"],
            "wqm-store": ["wqm-common"],
        },
        {},
        False,
        "are free of",
    ),
    (
        "B: transitively, too -- one hop away is still a link",
        {
            "wqm-common": [],
            "wqm-conventions": ["wqm-common"],
            "wqm-store": ["wqm-store-write"],
            "wqm-store-write": ["wqm-conventions"],
        },
        {},
        True,
        "links ['wqm-conventions']",
    ),
    (
        "C: wqm-sensor links a crate outside its allow-list",
        {
            "wqm-common": [],
            "wqm-conventions": ["wqm-common"],
            "wqm-sensor": ["wqm-conventions"],
        },
        {},
        True,
        "outside its declared closure",
    ),
    (
        "C: wqm-sensor limited to wqm-common passes",
        {"wqm-common": [], "wqm-sensor": ["wqm-common"]},
        {},
        False,
        "wqm closure is within",
    ),
    (
        "D: a storage crate links an algorithm crate (DP-7 wrong direction)",
        {
            "wqm-common": [],
            "wqm-graph": ["wqm-common"],
            "wqm-store": ["wqm-graph"],
        },
        {},
        True,
        "links algorithm crate(s): ['wqm-graph']",
    ),
    (
        "D: transitively, too -- the seam is a closure rule, not an edge rule",
        {
            "wqm-common": [],
            "wqm-intel": ["wqm-common"],
            "wqm-search": ["wqm-intel"],
            "wqm-store": ["wqm-search"],
        },
        {},
        True,
        "links algorithm crate(s): ['wqm-intel']",
    ),
    (
        "D: the LEGAL direction passes -- an algorithm crate may link its storage owner",
        {
            "wqm-common": [],
            "wqm-store": ["wqm-common"],
            "wqm-graph": ["wqm-store"],
        },
        {},
        False,
        "reach no algorithm crate",
    ),
    (
        "D: wqm-store-write may not exceed its §9.1 exact closure",
        {
            "wqm-common": [],
            "wqm-secrets": ["wqm-common"],
            "wqm-store-write": ["wqm-secrets"],
        },
        {},
        True,
        "exceeds its §9.1 exact closure",
    ),
    (
        "D: wqm-store-write within its exact closure passes",
        {
            "wqm-common": [],
            "wqm-store": ["wqm-common"],
            "wqm-search": ["wqm-store"],
            "wqm-store-write": ["wqm-search"],
        },
        {},
        False,
        "Exact closures enforced",
    ),
    (
        # Every rule in this file is about what a SHIPPED artifact links, and a
        # dev-dependency is compiled only by `cargo test` (P04-GT001-WO013). A
        # client bin whose integration test drives an S1 binary as a subprocess is
        # the concrete case: nothing S1-only enters the shipped executable.
        "A: a DEV-dependency on an S1-only crate is not a shipped link",
        {"wqm-common": [], "wqm-store-write": ["wqm-common"]},
        {"workspace-qdrant-mcp": ["wqm-common", "dev:wqm-store-write"]},
        False,
        "respect their closures",
    ),
    (
        # The converse, so the case above is a narrowing and not a hole.
        "A: the same crate as a NORMAL dependency still fails",
        {"wqm-common": [], "wqm-store-write": ["wqm-common"]},
        {"workspace-qdrant-mcp": ["wqm-common", "wqm-store-write"]},
        True,
        "links S1-only crate(s)",
    ),
    (
        "B: a kernel crate's DEV-dependency on wqm-conventions is not a link",
        {
            "wqm-common": [],
            "wqm-conventions": ["wqm-common"],
            "wqm-store": ["wqm-common", "dev:wqm-conventions"],
        },
        {},
        False,
        "PASS",
    ),
]


def main() -> int:
    failures = []
    for name, libs, bins, expect_fail, needle in CASES:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_workspace(root, libs, bins)
            code, output = run_guard(root)

        got_fail = code != 0
        if got_fail != expect_fail:
            failures.append(
                f"  {name}\n    expected {'FAIL' if expect_fail else 'PASS'}, "
                f"got exit {code}\n    output: {output.strip()}"
            )
            continue
        if needle not in output:
            failures.append(
                f"  {name}\n    exit code correct ({code}) but the diagnosis never "
                f"said {needle!r}\n    output: {output.strip()}"
            )
            continue
        print(f"  ok  {name}")

    if failures:
        sys.stderr.write(
            "selftest_link_closure: FAIL -- %d case(s) did not behave as claimed:\n%s\n"
            % (len(failures), "\n".join(failures))
        )
        return 1

    print(f"selftest_link_closure: PASS -- all {len(CASES)} cases behaved as claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
