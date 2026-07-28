#!/usr/bin/env python3
"""Name-registry single-producer guard -- N8 (ARCH rev15 §6.2).

N8 (crates/wqm-common/src/names/) is the single owner of the workspace's literal
identifier strings -- the four collection names and the five environment-variable
keys. Every other crate must source these through the N8 accessors, never re-spell
the literal. This guard greps the workspace `src/` trees for a stray re-spelling of
any guarded literal outside the N8 module and fails CI on a hit (the FP-2
one-producer rule).

Scope choices:
- Only high-value, distinctive literals are guarded (collection names, env keys).
  The N51 operation-class/consumer names ("read", "daemon", ...) are common English
  words that would trip on unrelated code; they are not guarded here.
- Each literal is matched only as a Rust string literal (`"projects"`), and only
  OUTSIDE comments -- see below.
- Only `src/` is scanned: `tests/` trees legitimately assert against the literal
  values, and the N8 module itself is the one legal producer.

Comments are stripped before matching, and that was a DEFECT FIX (`P04-GT001-WO012`).
This docstring previously claimed "prose in doc comments and identifiers do not trip
the guard" -- a claim that was simply untrue, because prose quoting a guarded name
(``a bare `"scratchpad"` would slip in unnoticed``) contains the very literal the
pattern looks for. It fired on a doc comment explaining why the write chokepoint
must not be bypassed.

This is the same defect `guard_no_skipped_tests` carried at `P04-GT001-WO007`, where
`#[ignore]` matched inside doc comments, and the same law applies: **a guard that
cannot be written about does not enforce a rule, it shapes prose** -- the natural
"fix" is to stop explaining the rule, which is worse than the guard's absence. The
cure there was matching over a stripped view while reporting the original line; the
cure was never swept to this sibling. It is now.

Note the asymmetry: this guard needs `literals=False`. A re-spelled name IS a string
literal, so blanking literals would blank the only thing it looks for. It strips
comments and keeps literals -- the exact opposite half of the same scan.

The guarded-literal list is the guard's own concern (what to police); extend it
when N8 grows (service/RPC names, table/column names, payload keys).

Exit 0 = clean, exit 1 = stray re-spelling found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from guard_codesize import strip_literals

RUST_ROOT = Path(__file__).resolve().parent.parent
N8_DIR = RUST_ROOT / "crates" / "wqm-common" / "src" / "names"

# The Phase-0 N8 vocabulary. Keep in step with crates/wqm-common/src/names/.
GUARDED = [
    "projects",
    "libraries",
    "rules",
    "scratchpad",
    "images",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "FASTEMBED_MODEL",
    "WQM_DATABASE_PATH",
    "WQM_LOG_LEVEL",
]


def guarded_pattern() -> re.Pattern[str]:
    alternation = "|".join(re.escape(lit) for lit in GUARDED)
    return re.compile(rf'"({alternation})"')


def policed_sources():
    for path in RUST_ROOT.rglob("*.rs"):
        parts = path.parts
        if "target" in parts or "tests" in parts:
            continue
        # N8 itself is the one legal producer.
        if N8_DIR in path.parents:
            continue
        yield path


def scan(src: str) -> list[tuple[int, str]]:
    """Return `(lineno, original_line)` for every guarded literal outside a comment.

    Matching runs over the comment-stripped view; REPORTING uses the original line,
    so a developer sees the code they wrote rather than a blanked skeleton. The
    stripper preserves newlines and byte offsets, so the two stay aligned.
    """
    pattern = guarded_pattern()
    stripped = strip_literals(src, literals=False).splitlines()
    original = src.splitlines()
    return [
        (lineno, original[lineno - 1])
        for lineno, line in enumerate(stripped, start=1)
        if pattern.search(line)
    ]


def main() -> int:
    hits: list[str] = []
    for path in policed_sources():
        for lineno, line in scan(path.read_text(errors="replace")):
            hits.append(f"{path.relative_to(RUST_ROOT)}:{lineno}: {line.strip()}")

    if hits:
        sys.stderr.write(
            "guard_name_registry: FAIL -- registry literal re-spelled outside N8 "
            f"({len(hits)} site(s)); source it through wqm-common::names (FP-2, N8):\n"
            + "\n".join(f"  {h}" for h in hits)
            + "\n"
        )
        return 1

    print(
        "guard_name_registry: PASS -- no guarded registry literal is re-spelled "
        "outside the N8 module."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
