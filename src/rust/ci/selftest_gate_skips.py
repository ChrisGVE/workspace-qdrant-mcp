#!/usr/bin/env python3
"""Prove the skip-set gate FAILS where it claims to.

`gate_skips.py` is the recovery floor under CR-007 defect (b), and the whole reason
that floor exists is that a green run is not evidence. That argument applies to the
gate itself: a gate nobody has seen fail is indistinguishable from one that cannot,
and this one runs on real CI logs that are usually empty.

The cases are chosen for the two directions the gate claims, plus the failure that
motivated direction 2 -- an empty log, which is what both "nothing skipped" and "the
sink is dead" look like from outside.

Exit 0 = every case behaved as claimed, exit 1 = a check did not bite.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
GATE = CI_DIR / "gate_skips.py"

POLICY = """
[[skip]]
test = "container_test"
condition = "no container runtime"
expected_on = ["macos-latest"]
allowed_on = ["some-byo-runner"]
run_locally = "cargo test -p wqm-test-harness --test container_round_trip"
"""


def run(log_lines: list[str], runner: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        log = root / "skips.txt"
        log.write_text("\n".join(log_lines) + ("\n" if log_lines else ""))
        policy = root / "policy.toml"
        policy.write_text(POLICY)
        out = subprocess.run(
            [
                sys.executable,
                str(GATE),
                "--log",
                str(log),
                "--runner",
                runner,
                "--policy",
                str(policy),
            ],
            capture_output=True,
            text=True,
        )
        return out.returncode, out.stdout + out.stderr


def run_without_log(runner: str) -> tuple[int, str]:
    """The gate against a log file that was never created at all."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        policy = root / "policy.toml"
        policy.write_text(POLICY)
        out = subprocess.run(
            [
                sys.executable,
                str(GATE),
                "--log",
                str(root / "absent.txt"),
                "--runner",
                runner,
                "--policy",
                str(policy),
            ],
            capture_output=True,
            text=True,
        )
        return out.returncode, out.stdout + out.stderr


SKIP = "WQM-TEST-SKIP: container_test -- no container runtime: docker absent"
STRAY = "WQM-TEST-SKIP: some_other_test -- the store was not reachable"

# (case name, callable, expect_fail, a substring the diagnosis must contain)
CASES = [
    (
        "1: an UNDECLARED skip fails, and is named",
        lambda: run([STRAY], "ubuntu-latest"),
        True,
        "UNDECLARED skip: some_other_test",
    ),
    (
        "1: a declared skip on a runner that allows it passes",
        lambda: run([SKIP], "some-byo-runner"),
        False,
        "declared skip: container_test",
    ),
    (
        "1: and the PASS publishes the command that runs it locally",
        lambda: run([SKIP], "some-byo-runner"),
        False,
        "run it locally: cargo test -p wqm-test-harness",
    ),
    (
        # The same skip, same policy, different runner. `ubuntu-latest` is on
        # neither list, so "declared somewhere" is not "declared here".
        "1: a skip declared for ANOTHER runner is still undeclared here",
        lambda: run([SKIP], "ubuntu-latest"),
        True,
        "UNDECLARED skip: container_test",
    ),
    (
        "2: an EXPECTED skip that did not appear fails",
        lambda: run([], "macos-latest"),
        True,
        "EXPECTED skip did not appear: container_test",
    ),
    (
        # The motivating case. An absent log is what a dead sink produces, and on
        # the positive-control runner it must not be green.
        "2: an ABSENT log file fails on the positive-control runner",
        lambda: run_without_log("macos-latest"),
        True,
        "the sink is broken",
    ),
    (
        # ...but on a runner with no expected skip, an empty log is the ordinary
        # good outcome and must NOT fail, or the gate cries wolf on every run.
        "2: an empty log on a runner with no expected skip passes",
        lambda: run([], "ubuntu-latest"),
        False,
        "0 skip(s) announced",
    ),
    (
        "2: the expected skip present on its own runner passes",
        lambda: run([SKIP], "macos-latest"),
        False,
        "declared skip: container_test",
    ),
    (
        # A line that is not an announcement must not be read as one -- prose in a
        # log is not a skip, and the marker is a contract with a shape.
        "3: an unrelated log line is not counted as a skip",
        lambda: run(
            ["some build output mentioning WQM-TEST-SKIP in passing"], "ubuntu-latest"
        ),
        False,
        "0 skip(s) announced",
    ),
]


def main() -> int:
    failures = []
    for name, case, expect_fail, needle in CASES:
        code, output = case()
        failed = code != 0
        if failed != expect_fail:
            failures.append(
                f"{name}: expected {'FAIL' if expect_fail else 'PASS'}, got exit {code}\n{output}"
            )
        elif needle not in output:
            failures.append(f"{name}: diagnosis missing {needle!r}\n{output}")
        else:
            print(f"  ok  {name}")

    if failures:
        sys.stderr.write("selftest_gate_skips: FAIL\n" + "\n".join(failures) + "\n")
        return 1
    print(f"selftest_gate_skips: PASS -- all {len(CASES)} cases behaved as claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
