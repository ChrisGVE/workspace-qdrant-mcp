//! The Service hub under a modal: the page is colourless, the box over it is not.
//!
//! VL §6 states the rule screen-wide, so it reaches this hub exactly as it reaches the
//! Dashboard and the Shell — Chris, 2026-09-07: *"we still have colors on the screen while all
//! should be muted (including the indicators)"*. This screen is the one that draws a real
//! [`Modal`], so it is also the only one where the *other* half of the rule is observable: the
//! scope covers the page and closes before anything painted over the stack.
//!
//! # Two guards, because the boundary has two sides and only one of them is on the box
//!
//! The first sweeps the page **outside** the modal's rectangle. It renders `Confirm`, which
//! carries no toast — the toast is deliberately outside the rule (§6: the must-see channel a
//! modal cannot suspend), so a frame carrying one would fail a sweep that is correct.
//!
//! The second is about **where the scope ends**, and it cannot be asked of the modal itself:
//! [`Modal`]'s whole vocabulary is neutral rungs — `normal` for the title, the body and each
//! action key, `muted` for the border and the action labels, a layer fill for the ground —
//! none of which this rule touches. A scope leaking into the box would therefore change nothing inside it, and a guard
//! written there could not fail. The toast is the surface above the stack that *does* carry a
//! hue ([`crate::widgets::toast`] paints [`Health::color`]), so that is where the drop point is
//! measured.

use super::*;
use crate::widgets::chrome::test_support::{coloured_cells, neutral_rungs};

/// The cells a sweep found, minus everything the modal covers.
fn outside(
    found: Vec<(u16, u16, &'static str, Color)>,
    rect: Rect,
) -> Vec<(u16, u16, &'static str, Color)> {
    found
        .into_iter()
        .filter(|(x, y, _, _)| {
            !((rect.x..rect.x + rect.width).contains(x)
                && (rect.y..rect.y + rect.height).contains(y))
        })
        .collect()
}

/// How many cells anywhere in `buf` carry `colour` as their foreground.
fn painted(buf: &Buffer, colour: Color) -> usize {
    (0..AREA.height)
        .flat_map(|y| (0..AREA.width).map(move |x| (x, y)))
        .filter(|(x, y)| buf.cell((*x, *y)).expect("cell in area").style().fg == Some(colour))
        .count()
}

/// Not one cell of the page behind the confirm carries a colour — and the confirm itself has
/// not been flattened along with it.
#[test]
fn the_page_under_a_service_modal_is_colourless_and_the_box_over_it_is_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let neutrals = neutral_rungs();
    let rect = frames::confirm_modal().rect(AREA);

    let confirm = render(frames::confirming());

    // The rectangle is asked of the modal rather than read off the render, so cross-check that
    // it is where the box actually landed — otherwise the sweep below could be excluding empty
    // space and passing over a coloured box.
    let (_, title_y) = find(&confirm, "Discard changes?").expect("the modal");
    assert!(
        (rect.y..rect.y + rect.height).contains(&title_y),
        "the modal drew at row {title_y}, outside the rect it reports ({rect:?})"
    );

    // The same hub with nothing over it: it has colour outside that rectangle, so the sweep is
    // measuring a page that had something to lose.
    let live = render(frames::base());
    assert!(
        !outside(coloured_cells(&live, &neutrals), rect).is_empty(),
        "the Service hub paints no colour even when it is live — this guard checks nothing"
    );

    let survivors = outside(coloured_cells(&confirm, &neutrals), rect);
    assert!(
        survivors.is_empty(),
        "{} cells kept a colour behind the modal, first ten: {:?}",
        survivors.len(),
        &survivors[..survivors.len().min(10)]
    );

    // And the box is not muted with the page. Asserted on its BODY, at the stated token: the
    // modal's border is `muted` and its action keys are `normal` whatever happens, so "some
    // cell inside the rect is not muted" would pass on a box nobody could read. The body is
    // the text the reader is actually being asked about.
    let (body_x, body_y) = find(&confirm, "watcher.debounce_ms").expect("the modal's body");
    assert!(
        (rect.x..rect.x + rect.width).contains(&body_x)
            && (rect.y..rect.y + rect.height).contains(&body_y),
        "the body drew at ({body_x}, {body_y}), outside the modal's own rect ({rect:?})"
    );
    assert_eq!(
        buf_fg(&confirm, body_x, body_y),
        Some(tokens::normal()),
        "the modal was flattened with the page beneath it — its body sits at `normal`"
    );
}

fn buf_fg(buf: &Buffer, x: u16, y: u16) -> Option<Color> {
    buf.cell((x, y)).expect("cell in area").style().fg
}

/// The scope closes before anything above the stack is drawn: a toast over the same confirm
/// keeps its own health hue while every reading on the page behind it has gone muted.
///
/// This is the guard on the **drop point**. Composed from two of this module's own frames
/// rather than invented — one store degraded, the toast that transition produces, and the
/// unsaved-edit confirm open over both — which is a screen the hub can genuinely be in.
#[test]
fn a_service_modal_leaves_the_toast_above_it_its_own_colour() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let yellow = Health::Degraded.color();
    let (deck, now) = frames::degraded_deck();

    let open = render(frames::degraded(&deck, now).modal(frames::confirm_modal()));
    let bare = render(frames::degraded(&deck, now));

    assert!(
        painted(&open, yellow) > 0,
        "the toast lost its hue — the page scope outlived the page it was opened for"
    );
    assert!(
        painted(&bare, yellow) > painted(&open, yellow),
        "the page behind the modal kept a degraded hue: {} lit cells with the modal open, \
         {} without it",
        painted(&open, yellow),
        painted(&bare, yellow)
    );
}
