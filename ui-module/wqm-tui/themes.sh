#!/usr/bin/env bash
#
# The bundled themes, as frames rather than as a list of names.
#
# Companion to palette.sh: that one answers "what does OUR ladder resolve to", this one
# answers "what would a bundled theme give us". Run it from a real terminal — the sheets are
# entirely about colour, so a pipe with no tty tells you nothing you can judge.
#
#   ./themes.sh                 # the gallery, then every theme's full sheet
#   ./themes.sh gallery         # just the gallery
#   ./themes.sh "Dracula"       # one theme's sheet
#
# Requires the instrument feature, which is NOT enabled in a shipping build:
#   cargo pantry dump ... --features themes-preview   (handled below)
set -u

WIDTH=${WIDTH:-112}
# Supplying the endpoints by hand keeps the surrounding chrome reproducible; the swatches
# are absolute RGB from theme data and do not depend on them.
export WQM_TUI_TERM_BG=${WQM_TUI_TERM_BG:-#1e1e2e}
export WQM_TUI_TERM_FG=${WQM_TUI_TERM_FG:-#cdd6f4}

SHEETS=("Catppuccin Mocha" "Catppuccin Latte" "Dracula" "Gruvbox dark, medium" "Solarized Light")

# NOTE: `cargo pantry dump --features X` forwards `--features` to the EXAMPLE BINARY, not to
# cargo, so it is silently ignored and anything gated behind an optional feature simply does
# not appear — an empty result that looks exactly like "no such entry". The example is
# therefore driven directly, which is also the only place the feature requirement is stated.
dump() {
	cargo run -q --example widget_preview \
		--features tui-pantry,themes-preview \
		-- --dump "Theme Sources" --variant "$1" --size "${WIDTH}x${2}" 2>/dev/null
}

case "${1:-all}" in
gallery)
	dump "Gallery — all 15" 24
	;;
all)
	echo
	echo "════ GALLERY — every theme ratatui-themes carries ════"
	dump "Gallery — all 15" 24
	# Then the per-theme detail: both representations of one theme, every swatch carrying
	# the name its own source gives it. Five schemes, chosen to disagree with each other:
	# Mocha names all sixteen slots, Solarized names none, Dracula's base06 equals base05,
	# Latte is the light polarity, Gruvbox is the well-behaved case.
	for sheet in "${SHEETS[@]}"; do
		echo
		echo "════ ${sheet} ════"
		dump "$sheet" 36
	done
	;;
*)
	dump "$1" 36
	;;
esac
