#!/usr/bin/env python3
"""Prove the name-registry guard bites real re-spellings and only those.

`guard_name_registry` was the last guard in this workspace with NO selftest, and
at `P04-GT001-WO012` that showed: its docstring claimed "prose in doc comments and
identifiers do not trip the guard", and it then failed the build on a doc comment
that quoted a guarded name while explaining why the write chokepoint must not be
bypassed. The claim had never been checked because nothing checked it.

That is the same defect `guard_no_skipped_tests` carried at `P04-GT001-WO007`, and
the same law: **a guard that cannot be written about does not enforce a rule, it
shapes prose.** The cure — match over a stripped view, report the original line —
existed in the tree already and had never been swept to this sibling.

Both directions are cases here. A guard that misses a real re-spelling is useless;
one that fires on prose is something people route around, and routing around this
one means deleting the sentence that explains N8.

Exit 0 = every case behaved as claimed, exit 1 = one did not.
"""

from __future__ import annotations

import sys

from guard_name_registry import scan

# (name, source, expected number of hits)
CASES: list[tuple[str, str, int]] = [
    (
        "a real re-spelling in code is caught",
        'const C: &str = "scratchpad";\n',
        1,
    ),
    (
        "an env-key re-spelling is caught",
        'let k = "QDRANT_URL";\n',
        1,
    ),
    (
        "a DOC comment quoting a guarded name does not fire",
        '//! a bare `"scratchpad"` would slip in unnoticed\n',
        0,
    ),
    (
        "a LINE comment quoting a guarded name does not fire",
        '// we must never write "projects" here\n',
        0,
    ),
    (
        "a BLOCK comment quoting a guarded name does not fire",
        '/* the "libraries" collection is N8-owned */\n',
        0,
    ),
    (
        "a NESTED block comment quoting a guarded name does not fire",
        '/* outer /* inner "rules" */ still comment */\n',
        0,
    ),
    (
        "an identifier that merely contains the word does not fire",
        "fn scratchpad_fixture() {}\n",
        0,
    ),
    (
        "a re-spelling on the SAME line as a comment is still caught",
        'let c = "scratchpad"; // the comment does not shield it\n',
        1,
    ),
    (
        "a re-spelling AFTER a comment mentioning it is still caught",
        '// never write "scratchpad" as a literal\nconst C: &str = "scratchpad";\n',
        1,
    ),
    (
        "a guarded name inside a STRING is caught, comment-like content and all",
        'let s = "scratchpad"; let t = "// not a comment";\n',
        1,
    ),
    (
        "the reported line is the ORIGINAL, not the stripped view",
        'const C: &str = "scratchpad"; // trailing\n',
        1,
    ),
    (
        "an unguarded word is not caught",
        'let s = "notebook";\n',
        0,
    ),
]


def main() -> int:
    failures = 0
    for name, src, expected in CASES:
        hits = scan(src)
        ok = len(hits) == expected
        # The last case additionally pins that reporting un-strips the line.
        if ok and name.startswith("the reported line"):
            ok = "// trailing" in hits[0][1]
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
        if not ok:
            failures += 1
            sys.stderr.write(
                f"    expected {expected} hit(s), got {len(hits)}: {hits!r}\n"
            )

    if failures:
        sys.stderr.write(
            f"selftest_name_registry: FAIL -- {failures} case(s) did not behave as claimed.\n"
        )
        return 1
    print(f"selftest_name_registry: PASS -- all {len(CASES)} cases behaved as claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
