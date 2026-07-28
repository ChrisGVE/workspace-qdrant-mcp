#!/usr/bin/env python3
"""Prove the no-skipped-tests guard bites real `#[ignore]`s and only those.

This guard had bitten before, so unlike the others it was never suspected of being
inert. What it had never been shown is the other half: that it does **not** fire on
text which merely *mentions* the thing it bans.

At `P04-GT001-WO007` it did exactly that — the test harness documents why `#[ignore]`
is not an escape from CR-007's discipline, and the guard failed the build on that
sentence. A guard that cannot be written about does not enforce a rule; it shapes
prose, and the natural fix is to stop explaining the rule.

Both directions are therefore cases here. False negatives make a guard useless; false
positives make it something people route around.

Exit 0 = every case behaved as claimed, exit 1 = one did not.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
GUARD = CI_DIR / "guard_no_skipped_tests.py"


def run_guard(root: Path) -> tuple[int, str]:
    # The guard resolves its scan root from its own location, so it is copied beside
    # a fixture tree rather than pointed at one.
    ci = root / "ci"
    ci.mkdir(parents=True, exist_ok=True)
    for name in ("guard_no_skipped_tests.py", "guard_codesize.py"):
        (ci / name).write_text((CI_DIR / name).read_text())
    out = subprocess.run(
        [sys.executable, str(ci / "guard_no_skipped_tests.py")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    return out.returncode, out.stdout + out.stderr


# (case name, rust source, expect_fail)
CASES = [
    ("a real #[ignore] is caught", "#[test]\n#[ignore]\nfn t() {}\n", True),
    (
        'a real #[ignore = "reason"] is caught',
        '#[test]\n#[ignore = "slow"]\nfn t() {}\n',
        True,
    ),
    (
        "a cfg_attr ignore is caught",
        '#[cfg_attr(target_os = "macos", ignore)]\nfn t() {}\n',
        True,
    ),
    ("a clean test file passes", "#[test]\nfn t() { assert!(true); }\n", False),
    # The WO007 regression, in all three shapes text can take.
    (
        "a DOC comment discussing #[ignore] does not fire",
        "//! `#[ignore]` is not an escape from the TDD charter.\nfn t() {}\n",
        False,
    ),
    (
        "a LINE comment discussing #[ignore] does not fire",
        "// we deliberately do not use #[ignore] here\nfn t() {}\n",
        False,
    ),
    (
        "a BLOCK comment discussing #[ignore] does not fire",
        "/* #[ignore] would be wrong here */\nfn t() {}\n",
        False,
    ),
    (
        "a string literal containing #[ignore] does not fire",
        'fn t() { let s = "#[ignore]"; let _ = s; }\n',
        False,
    ),
    # The failure mode a comment-stripping fix could introduce: code that FOLLOWS a
    # comment mentioning the attribute must still be scanned.
    (
        "a real #[ignore] AFTER a comment mentioning it is still caught",
        "// see #[ignore] policy\n#[test]\n#[ignore]\nfn t() {}\n",
        True,
    ),
]


def main() -> int:
    failures = []
    for name, source, expect_fail in CASES:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "crates" / "fixture" / "src"
            src.mkdir(parents=True)
            (src / "lib.rs").write_text(source)
            code, output = run_guard(root)

        if (code != 0) != expect_fail:
            failures.append(
                f"  {name}\n    expected {'FAIL' if expect_fail else 'PASS'}, "
                f"got exit {code}\n    output: {output.strip()}"
            )
            continue
        print(f"  ok  {name}")

    if failures:
        sys.stderr.write(
            "selftest_no_skipped_tests: FAIL -- %d case(s) did not behave as claimed:\n%s\n"
            % (len(failures), "\n".join(failures))
        )
        return 1

    print(
        f"selftest_no_skipped_tests: PASS -- all {len(CASES)} cases behaved as claimed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
