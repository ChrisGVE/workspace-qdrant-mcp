//! What the zone heading is pinned to.
//!
//! A sibling file rather than an inline `mod tests`, following `views::dashboard` and
//! `panes::cell`: the widget and the measurements taken against it are two readings, and
//! keeping them in one file is what pushed this module against the 500-line limit.

use super::*;
use crate::widgets::chrome::test_support::{render, row, style_at, Restore};

#[test]
fn a_screen_with_no_focused_zone_dims_nothing_and_accents_nothing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    // §3's third row. Two headings are rendered so the assertion is about the SCREEN:
    // with one heading, "nothing is dimmed" is vacuous — there is nothing to dim it
    // relative to.
    for index in 0..2 {
        let buf = render(ZoneHeading::new("Status", index, Attention::None));
        let line = row(&buf, 0);
        assert!(!line.contains(FOCUS_BAR), "no accent: {line:?}");
        assert_eq!(
            style_at(&buf, 0).fg,
            Some(tokens::normal()),
            "no zone dimmed on a screen with no focus"
        );
        assert!(!style_at(&buf, 0).add_modifier.contains(Modifier::BOLD));
    }
}

#[test]
fn focusing_one_zone_accents_it_and_recedes_the_other() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let attention = Attention::Zone(1);

    let live = render(ZoneHeading::new("Config", 1, attention));
    let line = row(&live, 0);
    assert!(line.starts_with("▌ Config"), "{line:?}");
    assert!(
        style_at(&live, 0).add_modifier.contains(Modifier::BOLD),
        "the focused heading is bold"
    );

    let receded = render(ZoneHeading::new("Status", 0, attention));
    assert!(!row(&receded, 0).contains(FOCUS_BAR));
    assert_eq!(style_at(&receded, 0).fg, Some(tokens::muted()));
}

#[test]
fn taking_focus_shifts_the_heading_text_two_columns_right() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    // The open question, stated as a measurement rather than as prose. If the prefix ever
    // becomes a gutter this test is what fails, and it fails saying exactly what changed.
    let idle = row(&render(ZoneHeading::new("Config", 1, Attention::None)), 0);
    let live = row(
        &render(ZoneHeading::new("Config", 1, Attention::Zone(1))),
        0,
    );

    // Counted in CHARACTERS, not bytes: `▌` is three bytes wide and one column wide, and
    // a byte offset would report a three-column shift the screen does not have.
    let column_of_heading = |line: &str| {
        line.chars()
            .position(|c| c == 'C')
            .expect("the heading is drawn")
    };
    let idle_x = column_of_heading(&idle);
    let live_x = column_of_heading(&live);
    assert_eq!(idle_x, 0);
    assert_eq!(
        live_x - idle_x,
        2,
        "the accent is a PREFIX (r02), so focus moves the text: {idle:?} / {live:?}"
    );
}

/// The accent is ONE letter — the key — and every other cell of the heading keeps the
/// heading's own rung. A title whose key is not its initial is used deliberately, so the
/// "before" half of the split is not empty and can be checked.
#[test]
fn the_key_letter_is_accented_and_nothing_beside_it_is() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(ZoneHeading::new("Last Errors", 0, Attention::Zone(0)).hotkey('e'));
    let line = row(&buf, 0);
    // `▌ Last Errors` — counted in CHARACTERS, since `▌` is three bytes and one column.
    let x = line
        .chars()
        .position(|c| c == 'E')
        .expect("the key letter is drawn") as u16;
    assert_eq!(
        style_at(&buf, x).fg,
        Some(tokens::accent()),
        "the key letter carries the accent: {line:?}"
    );
    // A focused zone is bold, key letter included — and the letter is bold on its own account
    // too, which the default-heading guard below is what actually measures.
    assert!(style_at(&buf, x).add_modifier.contains(Modifier::BOLD));
    assert!(
        !style_at(&buf, x).add_modifier.contains(Modifier::UNDERLINED),
        "no underline — the tab bar's digits carry none either"
    );

    for neighbour in [x - 1, x + 1] {
        assert_eq!(
            style_at(&buf, neighbour).fg,
            Some(tokens::normal()),
            "column {neighbour} is heading text, not the key: {line:?}"
        );
    }
}

/// VL §6: under a modal the page drops every highlight. The key letter goes muted, and the
/// `▌` and the weight — structure, not highlight — do not move.
#[test]
fn a_modal_mutes_the_key_letter_and_leaves_the_bar_and_the_weight() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = render(ZoneHeading::new("Rules (0)", 1, Attention::Zone(1)).hotkey('r'));
    let under = {
        let _modal = tokens::ModalScope::enter();
        render(ZoneHeading::new("Rules (0)", 1, Attention::Zone(1)).hotkey('r'))
    };

    let line = row(&live, 0);
    let x = line
        .chars()
        .position(|c| c == 'R')
        .expect("the key letter is drawn") as u16;
    assert_eq!(style_at(&live, x).fg, Some(tokens::accent()));
    assert_eq!(style_at(&under, x).fg, Some(tokens::muted()));

    assert_eq!(row(&under, 0), line, "a modal moves nothing");
    assert!(row(&under, 0).starts_with(&format!("{FOCUS_BAR} ")));
    assert!(
        style_at(&under, x).add_modifier.contains(Modifier::BOLD),
        "the focused zone keeps its weight under a modal"
    );
}

/// A key that is not in the title accents nothing at all. The honest frame: there is no
/// letter to press, so no letter is lit — and nothing panics looking for one.
#[test]
fn a_key_absent_from_the_title_accents_no_cell() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(ZoneHeading::new("Rules", 0, Attention::None).hotkey('z'));
    assert_eq!(row(&buf, 0).trim_end(), "Rules");
    for x in 0..crate::widgets::chrome::test_support::AREA.width {
        assert_ne!(
            style_at(&buf, x).fg,
            Some(tokens::accent()),
            "column {x} was accented for a key the title does not contain"
        );
    }
}

/// R1 (Chris, 2026-09-07: *"so that they are more visible on the screen"*): the key letter
/// carries BOTH the accent hue and bold, and the bold is the letter's own rather than the
/// heading's.
///
/// Measured on a heading at the DEFAULT rung, which is the only frame where the two can be
/// told apart: on a focused heading every cell is bold already, so a letter that had lost its
/// weight would still read as bold and the guard would pass on a broken screen.
#[test]
fn the_key_letter_is_bold_where_the_heading_around_it_is_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(ZoneHeading::new("Projects (29)", 0, Attention::None).hotkey('p'));
    let line = row(&buf, 0);
    let x = line
        .chars()
        .position(|c| c == 'P')
        .expect("the key letter is drawn") as u16;

    let key = style_at(&buf, x);
    assert_eq!(key.fg, Some(tokens::accent()), "hue: {line:?}");
    assert!(key.add_modifier.contains(Modifier::BOLD), "weight: {line:?}");
    assert!(
        !key.add_modifier.contains(Modifier::UNDERLINED),
        "bold was added, not underline: {line:?}"
    );

    // The cell AFTER the letter is ordinary heading text, and stays ordinary.
    let after = style_at(&buf, x + 1);
    assert!(
        !after.add_modifier.contains(Modifier::BOLD),
        "column {} is the `r` of Projects, not the key: {line:?}",
        x + 1
    );
    assert_eq!(after.fg, Some(tokens::normal()));
}

/// Under a modal the letter's HUE drops and its WEIGHT does not — the same split the selected
/// tab has always had (VL §6). Weight is structure; hue is highlight.
#[test]
fn a_modal_takes_the_key_letters_hue_and_leaves_its_weight() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let under = {
        let _modal = tokens::ModalScope::enter();
        render(ZoneHeading::new("Projects (29)", 0, Attention::None).hotkey('p'))
    };
    let style = style_at(&under, 0);
    assert_eq!(style.fg, Some(tokens::muted()), "the hue goes");
    assert!(
        style.add_modifier.contains(Modifier::BOLD),
        "the weight stays: {:?}",
        row(&under, 0)
    );
}

/// R6: with [`FocusMark::Block`] the live zone's heading is the tab line's inverse block, and
/// the key letter is gone — there is no letter left outside the block to accent.
#[test]
fn the_block_mark_inverts_the_whole_heading_and_drops_the_key() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        ZoneHeading::new("Rules (11)", 1, Attention::Zone(1))
            .hotkey('r')
            .focus_mark(FocusMark::Block),
    );
    let line = row(&buf, 0);
    assert!(line.starts_with(" Rules (11) "), "{line:?}");
    assert!(!line.contains(FOCUS_BAR), "the block replaces the bar: {line:?}");

    for x in 0.." Rules (11) ".chars().count() as u16 {
        assert_eq!(style_at(&buf, x).bg, Some(tokens::selector()), "column {x}");
        assert_ne!(
            style_at(&buf, x).fg,
            Some(tokens::accent()),
            "column {x} still lights a key inside the block: {line:?}"
        );
    }
}

/// Under a modal the block gives way to muted bold text, keeping its two spaces so nothing on
/// the row moves — exactly what the tab bar does with the selected tab (VL §6).
#[test]
fn a_modal_takes_the_blocks_fill_and_leaves_its_weight() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = render(
        ZoneHeading::new("Rules (11)", 1, Attention::Zone(1)).focus_mark(FocusMark::Block),
    );
    let under = {
        let _modal = tokens::ModalScope::enter();
        render(ZoneHeading::new("Rules (11)", 1, Attention::Zone(1)).focus_mark(FocusMark::Block))
    };

    assert_eq!(row(&under, 0), row(&live, 0), "a modal moves nothing");
    assert_eq!(style_at(&live, 1).bg, Some(tokens::selector()));
    assert_ne!(
        style_at(&under, 1).bg,
        Some(tokens::selector()),
        "the fill goes — a modal leaves no selector behind it"
    );
    assert_eq!(style_at(&under, 1).fg, Some(tokens::muted()));
    assert!(style_at(&under, 1).add_modifier.contains(Modifier::BOLD));
}

/// An UNFOCUSED zone is unaffected by the mark it would wear if it were live — same treatment,
/// same key letter, whichever mark it carries.
#[test]
fn the_mark_changes_nothing_about_a_zone_that_is_not_live() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    for attention in [Attention::None, Attention::Zone(1)] {
        let bar = render(ZoneHeading::new("Rules (11)", 0, attention).hotkey('r'));
        let block = render(
            ZoneHeading::new("Rules (11)", 0, attention)
                .hotkey('r')
                .focus_mark(FocusMark::Block),
        );
        assert_eq!(row(&bar, 0), row(&block, 0), "{attention:?}");
        for x in 0..crate::widgets::chrome::test_support::AREA.width {
            assert_eq!(style_at(&bar, x), style_at(&block, x), "{attention:?} column {x}");
        }
    }
}
