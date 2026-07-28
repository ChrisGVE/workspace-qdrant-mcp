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
"""

from __future__ import annotations

import sys
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
ANCHOR = "wqm-guard: mac-marker-setter"


def rust_sources():
    for path in RUST_ROOT.rglob("*.rs"):
        if "target" in path.parts:
            continue
        yield path


def main() -> int:
    hits: list[str] = []
    for path in rust_sources():
        for lineno, line in enumerate(
            path.read_text(errors="replace").splitlines(), start=1
        ):
            if ANCHOR in line:
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
