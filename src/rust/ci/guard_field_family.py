#!/usr/bin/env python3
"""Field-family completeness guard -- ARCH rev15 §5.4, N35.

Every persisted field family (ARCH §5.3) must have exactly one row in the N35
field-family registry declaring its {metadata_tier, migration_disposition}. A
missing row would let a family migrate under an unstated (defaulted) disposition --
the class of silent data loss this rebuild exists to prevent, so the registry is
DEFAULT-DENY: no row, no migration.

The registry is a compile-time-exhaustive Rust table (N35's slice P04-GT055); its match-exhaustive
coverage of the `Collection`/family enums is enforced by rustc. This CI guard is
the second line: it confirms the registry module exists and is wired once the
families land. On the Phase-0 skeleton the registry is not present yet, so the
guard passes vacuously and says so. When P04-GT055 lands, extend `REGISTRY_MARKER`
detection to compare declared families against §5.3.

Exit 0 = clean, exit 1 = registry present but incomplete.
"""

from __future__ import annotations

import sys
from pathlib import Path

RUST_ROOT = Path(__file__).resolve().parent.parent
COMMON_SRC = RUST_ROOT / "crates" / "wqm-common" / "src"
# The registry's presence is detected by this anchor, placed on the field-family
# table when P04-GT055 lands.
REGISTRY_MARKER = "wqm-guard: field-family-registry"


def main() -> int:
    if not COMMON_SRC.exists():
        print("guard_field_family: PASS (vacuous) -- wqm-common has no source yet.")
        return 0

    marker_sites = [
        f"{path.relative_to(RUST_ROOT)}"
        for path in COMMON_SRC.rglob("*.rs")
        if REGISTRY_MARKER in path.read_text(errors="replace")
    ]

    if not marker_sites:
        print(
            "guard_field_family: PASS (vacuous) -- the N35 field-family registry "
            "is not present yet (arrives at P04-GT055; anchor its table with "
            "`// wqm-guard: field-family-registry`)."
        )
        return 0

    # Registry present: rustc already enforces match-exhaustiveness.
    # This guard confirms the registry is single-sited; deeper §5.3 coverage
    # comparison is added alongside the registry data at P04-GT055.
    if len(marker_sites) > 1:
        sys.stderr.write(
            "guard_field_family: FAIL -- the field-family registry is declared at "
            f"{len(marker_sites)} sites; it must be single-sited (FP-2):\n"
            + "\n".join(f"  {s}" for s in marker_sites)
            + "\n"
        )
        return 1

    print(f"guard_field_family: PASS -- registry present at {marker_sites[0]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
