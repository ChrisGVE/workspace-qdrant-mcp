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
    let want = want.trim_end_matches('\n');
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

/// Every screen still draws the frame it drew before the chrome became one object.
///
/// The bare page is **excluded, and that exclusion is the whole design change**: Chris asked
/// for the main window to end in a hairline and a minimalist help line (2026-09-14), and the
/// bare page is the one screen that had neither. Its fixtures are captured all the same —
/// they are the before-picture the new foot is measured against in `views::page`'s own tests,
/// which state the difference instead of hiding it.
#[test]
fn every_screen_still_draws_the_frame_it_drew_before_the_page_composition() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let capturing = std::env::var_os("WQM_TUI_CAPTURE").is_some();
    for (name, got) in capture_all() {
        if name.starts_with("shell-") && !capturing {
            continue;
        }
        check(&name, &got);
    }
}
