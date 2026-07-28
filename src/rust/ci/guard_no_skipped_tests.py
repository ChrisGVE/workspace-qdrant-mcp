#!/usr/bin/env python3
"""No-skipped-tests guard -- TDD charter anti-drift tooth (b), 2026-07-06.

The charter forbids disabling, ignoring, skipping, or commenting-out a test to get
a green suite. A red suite halts forward work; only a root-cause fix is allowed,
never a test edit to force green. This guard is the mechanical tooth: it fails the
build on any Rust test that is ignored or conditionally compiled out.

Banned:
  * `#[ignore]` / `#[ignore = "..."]`
  * `#[cfg_attr(..., ignore)]`
  * `#[cfg(ignore)]` / `#[cfg(any(ignore, ...))]`

There is deliberately no allow-list escape hatch: a genuinely long-running test
belongs behind an explicit feature gate that CI still runs, not behind `#[ignore]`.
Changing this policy is a charter amendment (coder + one auditor sign-off), not a
per-test override.

Matching runs over a comment- and literal-stripped view of each file. Without that,
this guard fires on any file that merely *discusses* `#[ignore]` -- which the test
harness's own documentation of the ban does (found at P04-GT001-WO007, by exactly
that route). A guard that cannot be written about is a guard that shapes prose
instead of code.

Exit 0 = clean, exit 1 = a disabled test was found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from guard_codesize import strip_literals

RUST_ROOT = Path(__file__).resolve().parent.parent
PATTERNS = [
    (re.compile(r"#\[\s*ignore\b"), "#[ignore]"),
    (re.compile(r"cfg_attr\s*\([^)]*\bignore\b"), "#[cfg_attr(.., ignore)]"),
    (re.compile(r"#\[\s*cfg\s*\([^)]*\bignore\b"), "#[cfg(ignore)]"),
]


def rust_sources():
    for path in RUST_ROOT.rglob("*.rs"):
        if "target" in path.parts:
            continue
        yield path


def main() -> int:
    hits: list[str] = []
    for path in rust_sources():
        source = path.read_text(errors="replace")
        # Match against the stripped view so prose about `#[ignore]` cannot trip the
        # ban, but REPORT the original line -- a diagnosis quoting blanked-out text
        # would be unreadable. `strip_literals` preserves newlines, so the two views
        # stay line-aligned.
        original = source.splitlines()
        stripped = strip_literals(source).splitlines()
        for lineno, line in enumerate(stripped, start=1):
            for pattern, label in PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(RUST_ROOT)
                    shown = original[lineno - 1].strip()
                    hits.append(f"  {rel}:{lineno}: {label}  ->  {shown}")

    if hits:
        sys.stderr.write(
            "guard_no_skipped_tests: FAIL -- a disabled/ignored test was found. "
            "The TDD charter forbids skipping a test to get green; fix the root "
            "cause instead:\n" + "\n".join(hits) + "\n"
        )
        return 1

    print("guard_no_skipped_tests: PASS -- no ignored or disabled tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
