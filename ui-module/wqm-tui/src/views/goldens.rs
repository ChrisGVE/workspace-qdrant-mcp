//! Every screen's frame, frozen — so a change to how the chrome is ASSEMBLED can be told apart
//! from a change to what it draws.
//!
//! Task 8 replaces four hand-assembled chromes with one [`crate::views::page::Page`]. The
//! milestone is composition, not a visual change, so the claim that has to stay checkable is
//! *the same screens come out*. A claim of that shape cannot be checked after the fact: once
//! the four assemblies are gone there is nothing left to compare against. These fixtures are
//! therefore captured from the renderer as it stood **before** the composition landed, and
//! committed before a line of it moved.
//!
//! # Symbols, not styles
//!
//! A fixture holds the row strings and nothing else. Colour is already pinned, and pinned far
//! better than a dump could: each screen's sweep asserts which rungs a cell may take
//! ([`crate::widgets::chrome::test_support::coloured_cells`]) and its hue guards assert the
//! ones it must. What no existing guard covers is **geometry** — which row a thing lands on,
//! and how wide it ends up — and geometry is exactly what moving the chrome into one object is
//! able to break silently.
//!
//! # Capturing
//!
//! `WQM_TUI_CAPTURE=1 cargo test --all-features -p wqm-tui goldens` writes the fixtures.
//! Without the variable a missing fixture is a failure and never a fresh capture: a golden
//! that re-blesses itself when it cannot find its own baseline is a golden that agrees with
//! whatever it is shown.
//!
//! # Gated on the pantry feature, like [`crate::tier`]
//!
//! The frames are the pantry's own — `views::*::ingredient` is where a screen's fixture data
//! lives, and the director gates from those frames. Capturing anything else would freeze a
//! screen nobody looks at.

use std::path::PathBuf;

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use crate::widgets::chrome::test_support::Restore;

/// The storyboard's own geometry, and the small screen the status block collapses on.
const SIZES: [(u16, u16); 2] = [(125, 34), (80, 20)];

/// One buffer as text: a row per line, trailing blanks dropped so a fixture is diffable.
pub(crate) fn rows(buf: &Buffer) -> String {
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render `draw` at an exact size and hand back its text.
pub(crate) fn dump(width: u16, height: u16, draw: impl FnOnce(Rect, &mut Buffer)) -> String {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    draw(area, &mut buf);
    rows(&buf)
}

/// A dump as exactly `height` rows.
///
/// [`str::lines`] drops the trailing empty ones, and a bare page ends in thirteen of them — so
/// a comparison that indexed straight off `lines()` would be reading one screen against
/// another of a different height, and fall off the end of the shorter.
fn padded(text: &str, height: usize) -> Vec<&str> {
    let mut rows: Vec<&str> = text.strip_suffix('\n').unwrap_or(text).lines().collect();
    rows.resize(height, "");
    rows
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/views/goldens")
        .join(format!("{name}.txt"))
}

/// Compare `got` against the committed fixture, or write it under `WQM_TUI_CAPTURE`.
pub(crate) fn check(name: &str, got: &str) {
    let path = fixture(name);
    if std::env::var_os("WQM_TUI_CAPTURE").is_some() {
        std::fs::write(&path, format!("{got}\n")).expect("write the fixture");
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "{} is missing. A golden with no baseline cannot be created by running the test — \
             re-capture it deliberately with WQM_TUI_CAPTURE=1",
            path.display()
        )
    });
    // Exactly the one newline `check` writes, never `trim_end`: a frame whose last rows are
    // blank — the bare page's are — would otherwise lose them on the way back in, and the
    // fixture would silently be of a shorter screen.
    let want = want.strip_suffix('\n').unwrap_or(want.as_str());
    if want == got {
        return;
    }
    let first = want
        .lines()
        .zip(got.lines())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| want.lines().count().min(got.lines().count()));
    panic!(
        "{name} no longer draws the frame it was captured with.\nfirst difference on row \
         {first}:\n  was: {:?}\n  now: {:?}",
        want.lines().nth(first),
        got.lines().nth(first)
    );
}

/// The four screens, each under the name its fixture carries.
fn capture_all() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (width, height) in SIZES {
        out.push((
            format!("shell-{width}x{height}"),
            dump(width, height, |area, buf| {
                crate::views::shell::frames::dashboard().render(area, buf)
            }),
        ));
        out.push((
            format!("dashboard-{width}x{height}"),
            dump(width, height, |area, buf| {
                crate::views::dashboard::ingredient::populated().render(area, buf)
            }),
        ));
        out.push((
            format!("queue-{width}x{height}"),
            dump(width, height, |area, buf| {
                crate::views::queue::ingredient::queue(Default::default()).render(area, buf)
            }),
        ));
        out.push((
            format!("service-{width}x{height}"),
            dump(width, height, |area, buf| {
                crate::views::service::frames::base().render(area, buf)
            }),
        ));
    }
    out
}

/// The screens the milestone was not supposed to touch still draw what they drew.
///
/// Two are **excluded, and each exclusion is a design change with Chris's words behind it**.
/// The bare page gains the foot he asked for on 2026-09-14; the Service tab gains the product
/// name its selector row had quietly lost. Both are captured here all the same, because their
/// fixtures are the before-pictures the two tests below measure those changes against —
/// naming a difference is what tells it apart from a regression.
#[test]
fn every_screen_still_draws_the_frame_it_drew_before_the_page_composition() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let capturing = std::env::var_os("WQM_TUI_CAPTURE").is_some();
    for (name, got) in capture_all() {
        let changed = name.starts_with("shell-") || name.starts_with("service-");
        if changed && !capturing {
            continue;
        }
        check(&name, &got);
    }
}

/// The Service tab's selector row is the others' selector row, and the difference is one word.
///
/// It had lost the product name, and the loss is the clearest evidence there was for why one
/// composition was needed: the hub laid its own tab row out with a
/// [`Layout`](ratatui::layout::Layout) of its own and rendered a bare
/// [`TabBar`](crate::widgets::tab_bar::TabBar) into it, while the other three went through the
/// app bar and got `WQM TUI` in front of the tabs. Nobody noticed for as long as four screens
/// each drew their own, because nothing could: there was no place the claim *the first rows of
/// every tab are the same rows* was written down as code.
///
/// Every other row is asserted byte-identical, so what this reports is a one-word gain and not
/// a re-laid-out screen.
#[test]
fn the_service_tab_gains_the_product_name_its_selector_row_had_lost() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const HEIGHT: usize = 34;
    let before = padded(include_str!("goldens/service-125x34.txt"), HEIGHT);
    let dumped = dump(125, HEIGHT as u16, |area, buf| {
        crate::views::service::frames::base().render(area, buf)
    });
    let after = padded(&dumped, HEIGHT);

    let title = crate::widgets::chrome::app_bar::TITLE;
    assert!(
        !before[0].contains(title) && after[0].contains(title),
        "the selector row is what changed: {:?} -> {:?}",
        before[0],
        after[0]
    );
    assert_eq!(
        after[0]
            .trim_start()
            .strip_prefix(title)
            .map(str::trim_start),
        Some(before[0].trim_start()),
        "the tabs themselves are untouched — the title is simply in front of them now"
    );
    for y in 1..HEIGHT {
        assert_eq!(
            before[y], after[y],
            "row {y} moved, and only row 0 should have"
        );
    }
}

/// The one screen whose frame the milestone changes, and the exact size of the change.
///
/// Chris, 2026-09-14: the main window ends in *"the hairline and the minimalist help line"*.
/// The bare page had neither — it was the only screen that stopped at its content — so it
/// gains two rows, and the region it hands the placeholder is two rows shorter. Everything
/// above is untouched, and asserting that against the before-picture is what tells a design
/// change apart from a regression.
///
/// The second half matters as much as the first: the help line is not blank. A page with no
/// tab built still offers `? Help · q Quit`, which is the pair
/// [`crate::widgets::chrome::status_line::ALWAYS`] keeps when nothing else fits.
#[test]
fn the_bare_page_gains_a_foot_and_moves_nothing_above_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const HEIGHT: usize = 34;
    let before = padded(include_str!("goldens/shell-125x34.txt"), HEIGHT);
    let dumped = dump(125, HEIGHT as u16, |area, buf| {
        crate::views::shell::frames::dashboard().render(area, buf)
    });
    let after = padded(&dumped, HEIGHT);

    // The top decoration: the selector, the frame rule, the status frame and its hairline.
    let top = crate::views::page::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL;
    for y in 0..top as usize {
        assert_eq!(before[y], after[y], "row {y} of the top moved");
    }

    // The foot is new, and it is exactly a hairline and one help line.
    let rule = crate::widgets::chrome::rule::RULE.repeat(125);
    let hairline = HEIGHT - crate::views::page::FOOT_ROWS as usize;
    assert!(
        before[hairline].trim().is_empty() && before[hairline + 1].trim().is_empty(),
        "the bare page had no foot before this — these two rows were empty"
    );
    assert_eq!(
        after[hairline], rule,
        "the foot opens with an edge-to-edge hairline"
    );
    assert_eq!(
        after[hairline + 1].trim(),
        "? Help   q Quit",
        "and the minimalist help line is the pair that always survives"
    );

    // And the placeholder re-centres in the region it is now given, rather than staying put
    // over a foot that would have covered it.
    let was = before
        .iter()
        .position(|line| line.contains(super::shell::PLACEHOLDER));
    let now = after
        .iter()
        .position(|line| line.contains(super::shell::PLACEHOLDER));
    assert_eq!(
        (was, now),
        (Some(20), Some(19)),
        "two rows off the region moves its centre by one"
    );
}
