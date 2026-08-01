#!/usr/bin/env bash
#
# The palette, as frames rather than as an argument.
#
# Run this from a real terminal. Palette::Derived asks the terminal for its own background
# and foreground (OSC 11/10) at startup, so piping this somewhere without a tty silently
# falls back to the black-to-white ladder instead of yours. To preview a *different* theme
# without switching to it, supply the endpoints by hand:
#
#   WQM_TUI_TERM_BG=#1e1e2e WQM_TUI_TERM_FG=#cdd6f4 ./palette.sh
#
set -u

WIDTH=${WIDTH:-104}

# The lookup table first: what colours exist at all. Sixteen theme slots, then every
# neutral rung with the value it resolved to, per palette. A rung that lands on the same
# value as the rung above it is flagged in place -- that collapse is what rules a palette
# out, and it should not have to be spotted by eye.
echo
echo "════ REFERENCE ════"
cargo pantry dump "Palette Reference" --variant "ANSI 16" --size "${WIDTH}x16" 2>/dev/null
# Bundled is in the loop because Bundled is what ships (§15). The script predates it and
# listed the other three only, so the one palette a user actually receives was the one this
# instrument did not print.
for palette in Theme Indexed Derived Bundled; do
	echo
	cargo pantry dump "Palette Reference" --variant "Rungs: $palette" \
		--size "${WIDTH}x12" 2>/dev/null
done

# `Palette Sheet` used to follow -- the whole vocabulary in use, per palette, once per
# surface. It is gone (Chris, 20260801: "we can remove the Palette Sheets, we'll work on
# actual screens"), and with it the side-by-side palette comparison, which `Bundled` being
# the default settled. What the sheet was the only renderer of is the LAYER model: a rung
# that reads cleanly on layer 0 can be swallowed by a modal fill. That question now belongs
# to the screens -- `cargo pantry dump "Service"` and the Modal variants -- rather than to a
# sheet that draws every role at once.
