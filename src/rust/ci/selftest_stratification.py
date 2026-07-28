#!/usr/bin/env python3
"""Prove the §9.1 stratification guard FAILS where it claims to.

`guard_stratification.py` enforces a claim -- "every workspace crate's wqm-crate
dependencies point strictly DOWN this stratification" -- that on today's workspace
has almost nothing to check: two stratum-0 crates and no edge between them. Its
direction check reports PASS (vacuous) and says so. That leaves its teeth resting
entirely here.

The cases are chosen for the failures the check must catch that a bare "no cycles"
rule would NOT: an up-edge and a same-level edge are both perfectly acyclic and both
forbidden. Those are the shape round 2 of the architecture loop found twice
(R2-MF-B), and the reason §9.1 publishes a stratification instead of asserting
acyclicity.

Exit 0 = every case behaved as claimed, exit 1 = a check did not bite.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from selftest_link_closure import write_crate

CI_DIR = Path(__file__).resolve().parent
GUARD = CI_DIR / "guard_stratification.py"

POLICY = """
[strata]
s0 = ["wqm-common", "wqm-proto"]
s1 = ["wqm-store", "wqm-secrets"]
s2 = ["wqm-search"]
s3 = ["wqm-store-write"]
s4 = ["wqm-graph"]
unstratified = ["wqm-service-install"]
unstratified_reason = "CR-023 -- ARCH rev15 §9.1's stratification omits it."
test_support = ["wqm-test-harness"]
"""


def build_workspace(root: Path, libs: dict[str, list[str]]) -> None:
    (root / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = ["crates/*"]\n'
    )
    for name, deps in libs.items():
        write_crate(root, name, deps, is_bin=False)
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


# (case name, libs, expect_fail, a substring the diagnosis must contain)
CASES = [
    (
        "1: a crate in the workspace but in no stratum",
        {"wqm-common": [], "wqm-unknown-crate": ["wqm-common"]},
        True,
        "in no stratum and on no declared exception list",
    ),
    (
        "1: a crate on the unstratified list fails loudly, naming its CR",
        {"wqm-common": [], "wqm-service-install": ["wqm-common"]},
        True,
        "CR-023",
    ),
    (
        "2: an UP-edge is acyclic and still forbidden",
        {
            "wqm-common": [],
            "wqm-graph": ["wqm-common"],
            "wqm-store": ["wqm-graph"],  # stratum 1 depending on stratum 4
        },
        True,
        "not strictly down",
    ),
    (
        "2: a SAME-stratum edge is acyclic and still forbidden",
        {"wqm-common": [], "wqm-proto": ["wqm-common"]},
        True,
        "not strictly down",
    ),
    (
        "2: a strictly-down edge passes",
        {"wqm-common": [], "wqm-store": ["wqm-common"]},
        False,
        "point strictly down",
    ),
    (
        "2: a strictly-down chain across three strata passes",
        {
            "wqm-common": [],
            "wqm-store": ["wqm-common"],
            "wqm-search": ["wqm-store"],
            "wqm-store-write": ["wqm-search"],
        },
        False,
        "point strictly down",
    ),
    (
        "3: the clean workspace reports acyclic",
        {"wqm-common": [], "wqm-store": ["wqm-common"]},
        False,
        "is acyclic",
    ),
    (
        "4: a test-support crate does NOT trip the coverage check",
        {"wqm-common": [], "wqm-test-harness": ["wqm-common"]},
        False,
        "are test-support",
    ),
    (
        "4: but a shipped crate linking it does -- the exemption's obligation",
        {
            "wqm-common": [],
            "wqm-test-harness": ["wqm-common"],
            "wqm-store": ["wqm-test-harness"],
        },
        True,
        "links test-support",
    ),
    (
        "4: transitively, too",
        {
            "wqm-common": [],
            "wqm-test-harness": ["wqm-common"],
            "wqm-search": ["wqm-test-harness"],
            "wqm-store": ["wqm-common"],
        },
        True,
        "links test-support",
    ),
]


def main() -> int:
    failures = []
    for name, libs, expect_fail, needle in CASES:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_workspace(root, libs)
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
            "selftest_stratification: FAIL -- %d case(s) did not behave as claimed:\n%s\n"
            % (len(failures), "\n".join(failures))
        )
        return 1

    print(
        f"selftest_stratification: PASS -- all {len(CASES)} cases behaved as claimed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
