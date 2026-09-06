#!/usr/bin/env python3
"""MAC-marker single-setter guard -- ARCH rev15 §6.2 N5.

The rules drain-marker MAC (HMAC-SHA256 over the length-prefixed marker input) must
be written from exactly ONE site. A second setter would let a forged or
inconsistent marker pass the drain check. Because the setter's concrete name is not
fixed until N5's slice P04-GT044, the single legitimate site is tagged with a stable guard anchor
comment:

    // wqm-guard: mac-marker-setter

This guard counts those anchors and fails if more than one exists. Zero (Phase 0,
before P04-GT044) is a pass, logged as such. Exit 0 = clean, exit 1 = duplicate setter.

WHAT COUNTS AS AN ANCHOR, and why that had to be said (`P04-GT002-WO074`). The anchor
is itself a comment, so the sibling guards' cure -- match over a comment-stripped
view -- would strip the one thing this guard looks for. The first version therefore
counted the bare substring anywhere in a line, and a doc comment in `wqm-common`
that *explained* the anchor (in prose, in backticks) was adopted as the single
legitimate setter: `PASS -- single setter at crates/wqm-common/src/rules_write_cap.rs`.
Left alone, N5's real setter would have made two hits and sent someone hunting a
duplicate that does not exist. The rule is now stated precisely enough to be wrong:
an anchor is a whole line that is a PLAIN line comment (`//`, never `///` or `//!`)
whose entire content is the anchor text. Prose that quotes it -- in a doc comment, in
backticks, mid-sentence -- is legal and does not count. A guard that cannot be
written about does not enforce a rule, it shapes prose (guard_name_registry's law).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
ANCHOR = "wqm-guard: mac-marker-setter"

# A plain `//` line comment (the third character is NOT `/` or `!`, so doc comments
# are excluded) whose whole content is the anchor. Anchored at both ends so a
# sentence that merely contains the anchor text does not register.
ANCHOR_LINE = re.compile(r"^\s*//(?![/!])\s*" + re.escape(ANCHOR) + r"\s*$")


def rust_sources():
    for path in RUST_ROOT.rglob("*.rs"):
        if "target" in path.parts:
            continue
        yield path


def scan(src: str) -> list[int]:
    """Line numbers (1-based) of every anchor line in `src`, under the rule above."""
    return [
        lineno
        for lineno, line in enumerate(src.splitlines(), start=1)
        if ANCHOR_LINE.match(line)
    ]


def main() -> int:
    hits: list[str] = []
    for path in rust_sources():
        for lineno in scan(path.read_text(errors="replace")):
            hits.append(f"{path.relative_to(RUST_ROOT)}:{lineno}")

    if len(hits) > 1:
        sys.stderr.write(
            "guard_mac_single_setter: FAIL -- the MAC-marker setter anchor "
            f"appears {len(hits)} times; exactly one setter is allowed "
            "(ARCH N5):\n" + "\n".join(f"  {h}" for h in hits) + "\n"
        )
        return 1

    if not hits:
        print(
            "guard_mac_single_setter: PASS (vacuous) -- no MAC-marker setter yet "
            "(arrives at P04-GT044; tag its one site with `// wqm-guard: "
            "mac-marker-setter`)."
        )
    else:
        print(f"guard_mac_single_setter: PASS -- single setter at {hits[0]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
