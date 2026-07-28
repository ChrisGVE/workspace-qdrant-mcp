#!/usr/bin/env python3
"""De-dash single-producer guard -- ARCH rev15 §6.2 N3.

The `fts_key(key) -> DedashedToken` de-dash transform must have exactly ONE
definition: it is called by N2 on the write side and N41's FTS5 query concrete on
the read side, and two divergent implementations would silently split the index
key space (the FP-2 "one producer" rule). This guard counts `fn fts_key`
definitions across the workspace source and fails if there is more than one.

Zero definitions (before N3's slice P04-GT032) is a pass, logged as such. Exit 0 = clean,
exit 1 = duplicate producer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
# A definition site: `fn fts_key(` optionally preceded by visibility/async/const.
DEF = re.compile(r"\bfn\s+fts_key\s*\(")


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
            if DEF.search(line):
                hits.append(f"{path.relative_to(RUST_ROOT)}:{lineno}")

    if len(hits) > 1:
        sys.stderr.write(
            "guard_dedash_single_producer: FAIL -- `fts_key` has "
            f"{len(hits)} definitions; exactly one is allowed (FP-2, ARCH N3):\n"
            + "\n".join(f"  {h}" for h in hits)
            + "\n"
        )
        return 1

    if not hits:
        print(
            "guard_dedash_single_producer: PASS (vacuous) -- `fts_key` is not "
            "defined yet (arrives at P04-GT032)."
        )
    else:
        print(f"guard_dedash_single_producer: PASS -- single producer at {hits[0]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
