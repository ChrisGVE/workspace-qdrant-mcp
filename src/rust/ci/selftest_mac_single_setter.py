#!/usr/bin/env python3
"""Prove the MAC single-setter guard counts anchors and only anchors.

`guard_mac_single_setter` was the last guard in this workspace with NO selftest, and
at `P04-GT002-WO074` that showed the same way it showed for its two siblings: a doc
comment in `wqm-common` that explained the anchor in prose was counted as the one
legitimate setter. The claim "counts the anchor" had never been checked against
"and nothing else", because nothing checked it.

This guard cannot use the siblings' cure (match over a comment-stripped view): the
anchor IS a comment. So the rule is stated on the line's shape instead -- a plain
`//` line comment whose whole content is the anchor -- and both directions are
cases here: the anchor line is counted, two of them are counted twice (the FAIL the
guard exists for), and every way of merely writing ABOUT the anchor is not.

Exit 0 = every case behaved as claimed, exit 1 = one did not.
"""

from __future__ import annotations

import sys

from guard_mac_single_setter import scan

# (name, source, expected number of hits)
CASES: list[tuple[str, str, int]] = [
    (
        "the anchor line is counted",
        "fn set_marker() {\n    // wqm-guard: mac-marker-setter\n    todo\n}\n",
        1,
    ),
    (
        "the anchor line is counted with any indentation and trailing space",
        "\t\t//   wqm-guard: mac-marker-setter   \n",
        1,
    ),
    (
        "two anchor lines are two hits -- the FAIL this guard exists for",
        "// wqm-guard: mac-marker-setter\nfn a() {}\n// wqm-guard: mac-marker-setter\n",
        2,
    ),
    (
        "a DOC comment (///) quoting the anchor does not fire",
        "/// tag the one site with `// wqm-guard: mac-marker-setter`\n",
        0,
    ),
    (
        "an INNER doc comment (//!) quoting the anchor does not fire",
        "//! the guard anchor `wqm-guard: mac-marker-setter` counts setters\n",
        0,
    ),
    (
        "a doc comment that is EXACTLY the anchor text does not fire",
        "/// wqm-guard: mac-marker-setter\n",
        0,
    ),
    (
        "a plain comment that merely contains the anchor mid-sentence does not fire",
        "// see wqm-guard: mac-marker-setter for the single-setter rule\n",
        0,
    ),
    (
        "the anchor inside a string literal does not fire -- it is a comment marker",
        'const A: &str = "// wqm-guard: mac-marker-setter";\n',
        0,
    ),
    (
        "a block comment holding the anchor does not fire",
        "/* wqm-guard: mac-marker-setter */\n",
        0,
    ),
    (
        "a near-miss spelling does not fire",
        "// wqm-guard: mac-marker-setters\n",
        0,
    ),
]


def main() -> int:
    failed = 0
    for name, src, want in CASES:
        got = len(scan(src))
        ok = got == want
        print(f"  {'ok ' if ok else 'BAD'} {name}")
        if not ok:
            print(f"        wanted {want} hit(s), got {got}")
            failed += 1
    if failed:
        print(f"selftest_mac_single_setter: FAIL -- {failed} of {len(CASES)} cases")
        return 1
    print(
        f"selftest_mac_single_setter: PASS -- all {len(CASES)} cases behaved as claimed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
