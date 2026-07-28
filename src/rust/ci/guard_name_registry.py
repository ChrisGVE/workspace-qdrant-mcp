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
- Each literal is matched only as a Rust string literal (`"projects"`), so prose in
  doc comments and identifiers do not trip the guard.
- Only `src/` is scanned: `tests/` trees legitimately assert against the literal
  values, and the N8 module itself is the one legal producer.

The guarded-literal list is the guard's own concern (what to police); extend it
when N8 grows (service/RPC names, table/column names, payload keys).

Exit 0 = clean, exit 1 = stray re-spelling found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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


def main() -> int:
    pattern = guarded_pattern()
    hits: list[str] = []
    for path in policed_sources():
        for lineno, line in enumerate(
            path.read_text(errors="replace").splitlines(), start=1
        ):
            if pattern.search(line):
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
