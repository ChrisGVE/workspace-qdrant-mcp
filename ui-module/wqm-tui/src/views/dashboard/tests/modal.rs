//! The whole-frame sweep: under a modal, not one cell of the Dashboard carries a colour.
//!
//! **This is the guard the defect needed.** Every per-widget modal guard passed while the
//! screen was visibly wrong (Chris, 2026-09-07, looking at `Dashboard / Under modal`: *"we
//! still have colors on the screen while all should be muted (including the indicators)"*).
//! Each of them was true — the tab bar did drop its accent, the cell heading did drop its fill
//! — and none of them could say *and nothing else on the page painted*, because a widget guard
//! is written for a widget somebody thought about. The RAG discs, the queue's three counts, the
//! roll-up dot and the queue triples inside the cells were covered by no guard at all.
//!
//! So this one asserts over the **frame**, against the neutral ladder enumerated from the
//! tokens ([`crate::widgets::chrome::test_support::neutral_rungs`]). A widget written next year
//! is inside the assertion the day it is drawn.

use super::*;

/// Not one cell anywhere on the page carries a hue or a fill while a modal owns the input.
#[test]
fn no_cell_of_the_dashboard_carries_a_colour_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let neutrals = neutral_rungs();

    // The live frame first: a screen that painted nothing anyway would pass the sweep below
    // silently, and a guard that cannot fail is not a guard.
    let live = render(view(frames::populated()), WIDE, TALL);
    assert!(
        !coloured_cells(&live, &neutrals).is_empty(),
        "the Dashboard paints no colour even when it is live — this guard checks nothing"
    );

    let under = render(view(frames::populated()).under_modal(true), WIDE, TALL);
    let survivors = coloured_cells(&under, &neutrals);
    assert!(
        survivors.is_empty(),
        "{} cells kept a colour under a modal, first ten: {:?}",
        survivors.len(),
        &survivors[..survivors.len().min(10)]
    );
}

/// The focused frame too — the one that puts a selector block, a data cursor and a lit key
/// letter on the page at once, so the sweep above is not passing because the frame is quiet.
#[test]
fn a_focused_dashboard_under_a_modal_is_equally_colourless() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let neutrals = neutral_rungs();
    const RULES: usize = 3;

    let live = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    assert!(
        !coloured_cells(&live, &neutrals).is_empty(),
        "a focused Dashboard paints no colour — this guard checks nothing"
    );

    let under = render(
        view(frames::populated())
            .attention(Attention::Zone(RULES))
            .under_modal(true),
        WIDE,
        TALL,
    );
    let survivors = coloured_cells(&under, &neutrals);
    assert!(
        survivors.is_empty(),
        "{} cells kept a colour under a modal, first ten: {:?}",
        survivors.len(),
        &survivors[..survivors.len().min(10)]
    );
}
