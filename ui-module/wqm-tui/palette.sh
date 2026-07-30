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
for palette in Theme Indexed Derived; do
	echo
	cargo pantry dump "Palette Reference" --variant "Rungs: $palette" \
		--size "${WIDTH}x12" 2>/dev/null
done

# Then the vocabulary in use, all three palettes stacked: the tab selector against both
# alarm hues, the four emphasis rungs, the structural greys, the cursor and edit fills side
# by side, the layer backgrounds, and the health glyphs.
#
# Three times, once per surface. Everything above renders on the terminal's own background,
# which is layer 0 -- so on its own it never exercises the depth model at all. A rung that
# reads cleanly there can be swallowed by a modal fill, and that is only visible here.
for surface in "All Palettes" "All Palettes on Layer 1" "All Palettes on Layer 2"; do
	echo
	echo "════ ${surface^^} ════"
	cargo pantry dump "Palette Sheet" --variant "$surface" --size "${WIDTH}x40" 2>/dev/null
done
