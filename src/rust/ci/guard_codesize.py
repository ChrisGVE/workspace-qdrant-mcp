#!/usr/bin/env python3
"""Codesize guard -- ARCH rev15 §9 limits, enforced (Rust: 500 lines/file, 80/function).

§9 opens with the limits as a statement of fact ("Rust limits: 500 lines/file, 80
lines/function"), and the GT001 charter §5 lists codesize among the standing laws
that apply to every WO. Measured at P04-GT001-WO006, the predecessor `codesize.sh`
did neither thing it was credited with: it printed "report only (non-gating)" and it
never looked at function length at all -- half the limit was unmeasured, and the
measured half could not fail a build.

This guard gates both halves. It is free to adopt now precisely because the
workspace is small and clean; a limit adopted after the violations exist has to be
retrofitted against them (charter §5A.4).

Line counting is deliberately naive -- raw physical lines, comments and inline
`#[cfg(test)]` modules included. That is the same count `wc -l` gives and the same
count the limits were written against; a guard that discounts test code invites the
"it is only tests" exemption the limits exist to refuse.

Function length is measured from the line carrying `fn` through the line closing its
body. Bodiless declarations (trait signatures, `extern` blocks) have no length and
are skipped. Brace counting runs over a comment- and literal-stripped view of the
source, so a `{` inside a string or a doc comment cannot move the count.

Exit 0 = within limits, exit 1 = any file or function over, or a scan failure.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
FILE_LIMIT = 500
FN_LIMIT = 80

FN_START = re.compile(r"\bfn\s+[A-Za-z_][A-Za-z0-9_]*")


def strip_literals(src: str) -> str:
    """Return `src` with comments and literals blanked, newlines preserved.

    Every stripped character becomes a space so byte offsets -- and therefore line
    numbers -- stay identical to the original. Handles nested block comments, raw
    strings with any hash count, byte strings, and the `'a` lifetime vs `'a'` char
    ambiguity (a quote is a char literal only if a closing quote follows within the
    escape-aware window).
    """
    out = list(src)
    i, n = 0, len(src)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        ch = src[i]
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
            continue
        # Raw / byte strings: r"..", r#".."#, b"..", br#".."#
        m = re.match(r'(?:b?r(#*)"|b")', src[i:])
        if m:
            if m.group(0).startswith(("r", "br")):
                terminator = '"' + m.group(1)
                j = src.find(terminator, i + len(m.group(0)))
                j = n if j < 0 else j + len(terminator)
            else:
                j = _end_of_quoted(src, i + len(m.group(0)), '"')
            blank(i, j)
            i = j
            continue
        if ch == '"':
            j = _end_of_quoted(src, i + 1, '"')
            blank(i, j)
            i = j
            continue
        if ch == "'":
            j = _end_of_char_literal(src, i)
            if j is not None:
                blank(i, j)
                i = j
                continue
            i += 1  # a lifetime, not a literal
            continue
        i += 1

    return "".join(out)


def _end_of_quoted(src: str, start: int, quote: str) -> int:
    """Index just past the closing `quote`, honouring backslash escapes."""
    i, n = start, len(src)
    while i < n:
        if src[i] == "\\":
            i += 2
            continue
        if src[i] == quote:
            return i + 1
        i += 1
    return n


def _end_of_char_literal(src: str, start: int) -> int | None:
    """Index just past a char literal beginning at `start`, or None for a lifetime."""
    i = start + 1
    if i < len(src) and src[i] == "\\":
        end = _end_of_quoted(src, i, "'")
        return end if end <= len(src) else None
    if i + 1 < len(src) and src[i + 1] == "'":
        return i + 2
    return None


def function_spans(src: str) -> list[tuple[str, int, int]]:
    """Return (name, first_line, last_line) for every function with a body."""
    code = strip_literals(src)
    line_of = _line_index(code)
    spans: list[tuple[str, int, int]] = []

    for match in FN_START.finditer(code):
        # Walk forward to whichever comes first: the body's `{` or a `;`/`=`
        # that proves there is no body (trait signature, `extern` decl, type alias).
        i, n = match.end(), len(code)
        while i < n and code[i] not in "{;":
            i += 1
        if i >= n or code[i] == ";":
            continue
        depth, j = 0, i
        while j < n:
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise ValueError(f"unbalanced braces after `{match.group(0)}`")
        name = match.group(0).split()[-1]
        spans.append((name, line_of(match.start()), line_of(j)))
    return spans


def _line_index(text: str):
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)

    def line_of(offset: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    return line_of


def rust_sources(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.rs") if "target" not in p.relative_to(root).parts
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="codesize guard (files + functions)")
    ap.add_argument("--root", type=Path, default=RUST_ROOT)
    ap.add_argument("--file-limit", type=int, default=FILE_LIMIT)
    ap.add_argument("--fn-limit", type=int, default=FN_LIMIT)
    args = ap.parse_args(argv)

    files = rust_sources(args.root)
    if not files:
        sys.stderr.write(
            f"guard_codesize: FAIL -- no Rust sources found under {args.root}. "
            "A codesize guard that scans nothing is a guard that cannot fail.\n"
        )
        return 1

    over_files, over_fns, fn_count = [], [], 0
    for path in files:
        rel = path.relative_to(args.root)
        src = path.read_text(encoding="utf-8", errors="replace")
        lines = src.count("\n") + (0 if src.endswith("\n") or not src else 1)
        if lines > args.file_limit:
            over_files.append(f"  {lines:5d} lines  {rel} (limit {args.file_limit})")
        try:
            spans = function_spans(src)
        except ValueError as exc:
            sys.stderr.write(f"guard_codesize: FAIL -- {rel}: {exc}\n")
            return 1
        fn_count += len(spans)
        for name, first, last in spans:
            length = last - first + 1
            if length > args.fn_limit:
                over_fns.append(
                    f"  {length:5d} lines  {rel}:{first} `fn {name}` "
                    f"(limit {args.fn_limit})"
                )

    if over_files or over_fns:
        sys.stderr.write("guard_codesize: FAIL -- codesize limits exceeded.\n")
        if over_files:
            sys.stderr.write("files over the line limit:\n")
            sys.stderr.write("\n".join(over_files) + "\n")
        if over_fns:
            sys.stderr.write("functions over the line limit:\n")
            sys.stderr.write("\n".join(over_fns) + "\n")
        sys.stderr.write(
            "(ARCH rev15 §9: Rust 500 lines/file, 80 lines/function. Split by "
            "responsibility -- extract the section being changed into its own "
            "module rather than shaving lines.)\n"
        )
        return 1

    print(
        f"guard_codesize: PASS -- {len(files)} file(s) within {args.file_limit} lines, "
        f"{fn_count} function(s) within {args.fn_limit}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
