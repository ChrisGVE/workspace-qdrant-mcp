#!/usr/bin/env python3
"""Prove the codesize guard FAILS where it claims to.

`guard_codesize.py` passes on the real workspace and will keep passing for a while
-- the largest file is a fraction of the limit. A limit guard that has only ever
seen compliant input has never demonstrated it can measure anything at all, and its
predecessor `codesize.sh` is the cautionary case: it printed a green line for months
while never looking at function length and never being able to fail a build.

Two failure classes must bite (a file over 500, a function over 80) and one must
not: a long file made of short functions is legal, and a guard that cannot tell the
two apart is measuring the wrong thing.

The third group is the one that actually matters. Brace counting is only as good as
the literal stripping underneath it, so the fixtures hide `{` and `}` inside string
literals, raw strings with hashes, nested block comments, char literals, and next to
lifetime annotations. If any of those moved the count, a function's measured length
would be wrong in a direction nobody would notice.

Exit 0 = every case behaved as claimed, exit 1 = a case did not.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
GUARD = CI_DIR / "guard_codesize.py"

FILE_LIMIT = 10
FN_LIMIT = 5


def run_guard(root: Path, file_limit: int, fn_limit: int) -> tuple[int, str]:
    out = subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--root",
            str(root),
            "--file-limit",
            str(file_limit),
            "--fn-limit",
            str(fn_limit),
        ],
        capture_output=True,
        text=True,
    )
    return out.returncode, out.stdout + out.stderr


def body(lines: int) -> str:
    """A function body of exactly `lines` statement lines."""
    return "\n".join("    let _x = 1;" for _ in range(lines))


# A function whose braces are all hidden inside literals and comments. If the
# stripper leaks, the guard either mis-measures it or dies on unbalanced braces.
TRICKY = f"""fn tricky() -> &'static str {{
{body(1)}
    let s = "a {{ brace }} inside a string";
    let r = r#"raw {{ with "quotes" and }} braces"#;
    let b = b"bytes {{ }}";
    /* nested /* block */ comment with {{ braces }} */
    let c = '{{';
    let d = '\\'';
    let _ = (s, r, b, c, d);
    "ok"
}}
"""

# (case name, {filename: contents}, expect_fail, needle, (file_limit, fn_limit))
# Most cases run against the tiny default limits; the literal-stripping cases carry
# their own so the fixture's own size does not decide the outcome.
CASES = [
    (
        f"a file over {FILE_LIMIT} lines fails",
        {"big.rs": "// filler\n" * (FILE_LIMIT + 1)},
        True,
        "files over the line limit",
    ),
    (
        f"a file at exactly {FILE_LIMIT} lines passes",
        {"edge.rs": "// filler\n" * FILE_LIMIT},
        False,
        "within",
    ),
    (
        f"a function over {FN_LIMIT} lines fails, and is named",
        {"fn_over.rs": "fn too_long() {\n" + body(FN_LIMIT) + "\n}\n"},
        True,
        "`fn too_long`",
    ),
    (
        f"a function at exactly {FN_LIMIT} lines passes",
        {"fn_edge.rs": "fn just_fits() {\n" + body(FN_LIMIT - 2) + "\n}\n"},
        False,
        "within",
    ),
    (
        "a long file of short functions is legal -- the limits are independent",
        {
            "many.rs": "\n".join(
                f"fn f{i}() {{\n    let _x = {i};\n}}" for i in range(3)
            )
            + "\n"
        },
        False,
        "within",
    ),
    (
        "a bodiless trait signature has no length and is not counted",
        {"decl.rs": "trait T {\n    fn no_body(&self) -> u32;\n}\n"},
        False,
        "0 function(s)",
    ),
    # The next two bracket `fn tricky`'s measured length at EXACTLY 11 lines: it
    # fails at a limit of 10 naming that figure, and passes at 11. Any leak in the
    # literal stripping moves that number (or raises on unbalanced braces), so the
    # pair is a measurement assertion, not a pass/fail one.
    (
        "braces in strings/raw strings/nested comments/char literals: measured length",
        {"tricky.rs": TRICKY},
        True,
        "11 lines  tricky.rs:1 `fn tricky`",
        (100, 10),
    ),
    (
        "the same function passes at a limit of 11 -- the bracket closes",
        {"tricky.rs": TRICKY},
        False,
        "1 function(s) within 11",
        (100, 11),
    ),
    (
        "an empty tree fails rather than reporting a vacuous green",
        {},
        True,
        "no Rust sources found",
    ),
]


def main() -> int:
    failures = []
    for case in CASES:
        name, files, expect_fail, needle = case[:4]
        file_limit, fn_limit = case[4] if len(case) > 4 else (FILE_LIMIT, FN_LIMIT)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for filename, contents in files.items():
                (root / filename).write_text(contents)
            code, output = run_guard(root, file_limit, fn_limit)

        got_fail = code != 0
        if got_fail != expect_fail:
            failures.append(
                f"  {name}\n    expected {'FAIL' if expect_fail else 'PASS'}, "
                f"got exit {code}\n    output: {output.strip()}"
            )
            continue
        if needle not in output:
            failures.append(
                f"  {name}\n    exit code correct ({code}) but the diagnosis never "
                f"said {needle!r}\n    output: {output.strip()}"
            )
            continue
        print(f"  ok  {name}")

    if failures:
        sys.stderr.write(
            "selftest_codesize: FAIL -- %d case(s) did not behave as claimed:\n%s\n"
            % (len(failures), "\n".join(failures))
        )
        return 1

    print(f"selftest_codesize: PASS -- all {len(CASES)} cases behaved as claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
