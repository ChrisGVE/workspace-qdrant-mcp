#!/usr/bin/env bash
#
# The bundled themes, as frames rather than as a list of names.
#
# Companion to palette.sh: that one answers "what does OUR ladder resolve to", this one
# answers "what would a bundled theme give us". Run it from a real terminal — the sheets are
# entirely about colour, so a pipe with no tty tells you nothing you can judge.
#
#   ./themes.sh                 # the gallery — every theme, ten fields each
#
# Requires the instrument feature, which is NOT enabled in a shipping build:
#   cargo pantry dump ... --features themes-preview   (handled below)
set -u

WIDTH=${WIDTH:-112}
# Supplying the endpoints by hand keeps the surrounding chrome reproducible; the swatches
# are absolute RGB from theme data and do not depend on them.
export WQM_TUI_TERM_BG=${WQM_TUI_TERM_BG:-#1e1e2e}
export WQM_TUI_TERM_FG=${WQM_TUI_TERM_FG:-#cdd6f4}

# NOTE: `cargo pantry dump --features X` forwards `--features` to the EXAMPLE BINARY, not to
# cargo, so it is silently ignored and anything gated behind an optional feature simply does
# not appear — an empty result that looks exactly like "no such entry". The example is
# therefore driven directly, which is also the only place the feature requirement is stated.
dump() {
	cargo run -q --example widget_preview \
		--features tui-pantry,themes-preview \
		-- --dump "Theme Sources" --variant "$1" --size "${WIDTH}x${2}" 2>/dev/null
}

# The per-theme comparison sheets are gone with the base16 half they compared against
# (VISUAL-LANGUAGE §10 closed that question). The gallery is the instrument now.
dump "Gallery — all 15" 24
