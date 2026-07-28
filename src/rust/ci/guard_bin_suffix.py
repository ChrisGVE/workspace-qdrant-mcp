#!/usr/bin/env python3
"""Deployment-suffix guard for the binaries -- the knob's declared exception.

`PROJECT_LOGISTICS.md` (decided 2026-07-01) runs v0.2 as a parallel `-v2` system
alongside production, never in place, and CHARTER §2.4 routes four name families
through one knob while declaring the **binaries an exception**: a physical
filename cannot be built by `concat!`, so `[[bin]] name` is data in a manifest,
not a derived constant.

An exception without an enforced converse is a hole (P04-GT001-WO006). The
converse here is the requirement the exception exists to satisfy: **the binary
filenames must DIFFER from production's**, because `memexd` and `memexd` are the
same path in `/usr/local/bin` and the parallel deployment's whole premise is that
production keeps running. So this guard checks what the exception is exempt from
deriving: that every bin carries the suffix the knob is currently set to.

The suffix is EXTRACTED from `deployment_suffix!()` in
`crates/wqm-common/src/names/deployment.rs`, never restated here (MCP-SURFACE.md
§7.3a). Flipping that macro to `""` therefore retires this guard automatically:
after cutover every name trivially ends with the empty suffix, which is the
honest vacuity -- the parallel period is over and there is nothing left to
collide with.

Exit 0 = every bin carries the current deployment suffix, exit 1 = one does not.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_RS = RUST_ROOT / "crates" / "wqm-common" / "src" / "names" / "deployment.rs"
BINS_DIR = RUST_ROOT / "bins"

# The macro body, as written: `macro_rules! deployment_suffix { () => { "-v2" }; }`
SUFFIX_MACRO = re.compile(
    r"macro_rules!\s+deployment_suffix\s*\{\s*\(\s*\)\s*=>\s*\{\s*\"([^\"]*)\"",
    re.MULTILINE,
)


def deployment_suffix() -> str:
    """The one spelling of the knob, read from N8 rather than repeated."""
    source = DEPLOYMENT_RS.read_text()
    match = SUFFIX_MACRO.search(source)
    if match is None:
        sys.stderr.write(
            "guard_bin_suffix: FAIL -- could not extract `deployment_suffix!()` from "
            f"{DEPLOYMENT_RS.relative_to(RUST_ROOT)}. The knob moved or was renamed; "
            "this guard reads it rather than restating it, so it must be re-pointed.\n"
        )
        raise SystemExit(1)
    return match.group(1)


def declared_bins() -> list[tuple[Path, str]]:
    """Every `[[bin]] name` under bins/, with the manifest that declares it."""
    found: list[tuple[Path, str]] = []
    for manifest in sorted(BINS_DIR.glob("*/Cargo.toml")):
        data = tomllib.loads(manifest.read_text())
        for entry in data.get("bin", []):
            name = entry.get("name")
            if name:
                found.append((manifest, name))
    return found


def main() -> int:
    suffix = deployment_suffix()
    bins = declared_bins()

    if not bins:
        print(
            "guard_bin_suffix: PASS (vacuous) -- no binary declares a `[[bin]] name` "
            "yet. Bites as soon as one does; the first three land at P04-GT001."
        )
        return 0

    if suffix == "":
        print(
            f"guard_bin_suffix: PASS (vacuous) -- the deployment suffix is empty, so "
            f"cutover has happened and there is no production deployment left to "
            f"collide with ({len(bins)} bin(s) checked)."
        )
        return 0

    offenders = [(m, n) for m, n in bins if not n.endswith(suffix)]
    if offenders:
        sys.stderr.write(
            f"guard_bin_suffix: FAIL -- {len(offenders)} binary name(s) do not carry "
            f"the deployment suffix `{suffix}`, so installing them would overwrite "
            f"production's binary of the same name (PROJECT_LOGISTICS.md, parallel "
            f"deployment):\n"
            + "\n".join(
                f'  {m.relative_to(RUST_ROOT)}: [[bin]] name = "{n}" '
                f"(expected `{n}{suffix}`)"
                for m, n in offenders
            )
            + "\n"
        )
        return 1

    print(
        f"guard_bin_suffix: PASS -- all {len(bins)} binary name(s) carry the "
        f"deployment suffix `{suffix}`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
