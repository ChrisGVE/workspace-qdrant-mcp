#!/usr/bin/env python3
"""Gate on the SET of announced test skips -- CR-007 defect (b)'s recovery floor.

`crates/wqm-test-harness/src/skip.rs` states the floor this script is:

    The floor is CI gating on the *set* of announced skips rather than on a green run
    (`P04-GT001-WO014`). The defect this cures is not one skipped test -- it is a
    suite drifting into asserting nothing while staying green, and only a set
    comparison catches that.

Two directions, and the second is what stops the gate itself from passing vacuously:

  1. **Unexpected skip** -- a test announced a skip that `ci/skip-policy.toml` does
     not declare for this runner. FAIL, naming the test and its reason.
  2. **Missing expected skip** -- the policy declares a skip `expected_on` this
     runner and it did not appear. FAIL. An empty log is what a BROKEN SINK produces,
     so a gate that only checks direction 1 reports green when the announcement path
     is dead. One runner carrying a positive control turns every run into proof that
     harness -> `WQM_TEST_SKIP_LOG` -> gate still works.

The log is written by `wqm_test_harness::announce_skip` in the format

    WQM-TEST-SKIP: <test name> -- <reason>

and the marker is pinned by a test (`the_marker_is_stable`) because it is a contract
between two languages: changing it in Rust without changing it here would produce a
silently-unmatched grep, which is the same green-with-nothing-checked this gate exists
to prevent.

Exit 0 = the announced set matches the declared set, exit 1 = it does not.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY = CI_DIR / "skip-policy.toml"

# Kept in step with `wqm_test_harness::SKIP_MARKER`, which a Rust test pins.
SKIP_MARKER = "WQM-TEST-SKIP"
LINE = re.compile(rf"^{re.escape(SKIP_MARKER)}:\s*(?P<test>\S+)\s*--\s*(?P<reason>.*)$")


def announced(log: Path) -> dict[str, str]:
    """Every skip the run announced, as `{test: reason}`.

    A missing log file means no skip was announced, which is a legitimate outcome and
    not an error -- direction 2 is what makes that distinguishable from a dead sink.
    """
    if not log.exists():
        return {}
    found: dict[str, str] = {}
    for raw in log.read_text(errors="replace").splitlines():
        match = LINE.match(raw.strip())
        if match:
            found[match.group("test")] = match.group("reason")
    return found


def declared(policy: Path) -> list[dict]:
    with policy.open("rb") as fh:
        return tomllib.load(fh).get("skip", [])


def check(
    entries: list[dict], seen: dict[str, str], runner: str
) -> tuple[bool, list[str]]:
    """Compare the two sets. Returns (ok, report lines)."""
    expected = {e["test"] for e in entries if runner in e.get("expected_on", [])}
    permitted = expected | {
        e["test"] for e in entries if runner in e.get("allowed_on", [])
    }
    by_test = {e["test"]: e for e in entries}

    report, problems = [], []

    for test, reason in sorted(seen.items()):
        if test in permitted:
            report.append(f"  declared skip: {test} -- {reason}")
            local = by_test[test].get("run_locally")
            if local:
                report.append(f"      run it locally: {local}")
        else:
            problems.append(
                f"  UNDECLARED skip: {test} -- {reason}\n"
                f"      Either the test lost its ability to assert, or this is a real\n"
                f"      condition that belongs in ci/skip-policy.toml with the command\n"
                f"      that runs it locally. Silence is not one of the options."
            )

    for test in sorted(expected - set(seen)):
        problems.append(
            f"  EXPECTED skip did not appear: {test}\n"
            f"      On `{runner}` this skip is the positive control that proves the\n"
            f"      announcement path works. Its absence means either the condition\n"
            f"      changed (update ci/skip-policy.toml) or the sink is broken (the\n"
            f"      harness no longer writes WQM_TEST_SKIP_LOG). Both need a human;\n"
            f"      a green run here would tell you neither."
        )

    return not problems, report + problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path, help="WQM_TEST_SKIP_LOG path")
    ap.add_argument(
        "--runner", required=True, help="the CI runner label, e.g. macos-latest"
    )
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = ap.parse_args()

    if not args.policy.exists():
        sys.stderr.write(f"gate_skips: FAIL -- no policy at {args.policy}\n")
        return 1

    seen = announced(args.log)
    ok, lines = check(declared(args.policy), seen, args.runner)

    stream = sys.stdout if ok else sys.stderr
    verdict = "PASS" if ok else "FAIL"
    stream.write(
        f"gate_skips: {verdict} -- {len(seen)} skip(s) announced on `{args.runner}`, "
        f"checked against {args.policy.name}.\n"
    )
    for line in lines:
        stream.write(line + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
