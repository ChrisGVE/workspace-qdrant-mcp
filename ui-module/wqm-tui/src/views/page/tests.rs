//! What the page's frame is pinned to.
//!
//! These are the claims that only became checkable once one object drew the chrome. While
//! four screens each assembled their own, *the first rows of every tab are the same rows* was
//! a sentence about four pieces of code, and the only honest way to check it was to read all
//! four — which is exactly how the Service tab came to be missing the product name for weeks.
//!
//! Gated on the pantry feature for the same reason `views::goldens` is: the frames a guard
//! reads must be the frames Chris looks at, and those live in the `ingredient` modules.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use super::*;
use crate::panes::status_block::{self, Queue as QueueCounts, StatusBlock, ENTRY_LABELS};
use crate::tokens::Health;
use crate::views::goldens::{dump, rows as buffer_rows};
use crate::widgets::chrome::rule::RULE;
use crate::widgets::chrome::status_line::ALWAYS;
use crate::widgets::chrome::test_support::{coloured_cells, neutral_rungs, Restore};
use crate::widgets::chrome::Freshness;

const WIDE: u16 = 125;
const TALL: u16 = 34;

/// One system state, so three tabs drawing the same top can be compared for being the same.
///
/// The frames in the pantry deliberately differ — a healthy workspace on one tab, a backlog on
/// another — which is right for looking at and useless for this: two tops that differ because
/// their INPUTS differ prove nothing about whether the rows are drawn the same way.
fn block() -> StatusBlock {
    let entries: [Health; ENTRY_LABELS.len()] = [
        Health::Healthy,
        Health::Degraded,
        Health::Healthy,
        Health::Healthy,
    ];
    StatusBlock::new(
        status_block::rollup(entries[0], &entries[1..]),
        "v0.2.0",
        Freshness::new(
            std::time::Duration::from_secs(4),
            std::time::Duration::from_secs(60),
        ),
        entries,
        QueueCounts {
            pending: 11_236,
            in_progress: 4,
            failed: 3,
            health: Health::Degraded,
        },
    )
}

/// The frame alone, at a stated tab, with whatever body a caller draws into the region.
fn page(active: usize, with_status: bool, width: u16, height: u16) -> (String, Rect) {
    let mut region = Rect::ZERO;
    let text = dump(width, height, |area, buf| {
        let mut page = Page::new(active);
        if with_status {
            page = page.status(block());
        }
        for (key, label) in ALWAYS {
            page = page.hint(key, label);
        }
        region = page.draw(area, buf);
    });
    (text, region)
}

/// Runs of blank cells collapsed to one, so a row can be compared for CONTENT while the
/// selection's own padding is allowed to move.
///
/// The tab bar marks the selected tab by widening it — a space either side of its number and
/// name — so two tabs' selector rows are never byte-identical and never should be. What has to
/// be identical is everything else on the row: the product name, the tab order, the spellings.
fn squeezed(line: &str) -> String {
    line.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The claim the constant top has always made, now that one thing makes it.
///
/// Given one system state, the six rows above the view are the same six rows on every tab that
/// carries a status frame — byte for byte, the selected tab's own padding excepted.
#[test]
fn the_top_rows_are_the_same_rows_on_every_tab() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let tops: Vec<Vec<String>> = [
        crate::views::dashboard::DASHBOARD_TAB,
        crate::views::queue::QUEUE_TAB,
        4,
    ]
    .into_iter()
    .map(|tab| {
        page(tab, true, WIDE, TALL)
            .0
            .lines()
            .take((CONSTANT_ROWS + status_block::ROWS_FULL) as usize)
            .map(str::to_string)
            .collect()
    })
    .collect();

    // Row 0 is the selector, and only the selection may move on it.
    let selectors: Vec<String> = tops.iter().map(|top| squeezed(&top[0])).collect();
    assert!(
        selectors.windows(2).all(|pair| pair[0] == pair[1]),
        "the selector row differs by more than its selection: {selectors:#?}"
    );
    // The guard has to be capable of failing: three rows that were already identical would
    // pass it without the tab bar doing anything at all.
    assert!(
        tops[0][0] != tops[1][0],
        "no tab is marked as selected — this comparison is checking nothing"
    );

    // Rows 1 to 5 are the frame rule, the status frame and the hairline that closes it, and
    // those carry no per-tab information whatsoever.
    for y in 1..(CONSTANT_ROWS + status_block::ROWS_FULL) as usize {
        assert!(
            tops.windows(2).all(|pair| pair[0][y] == pair[1][y]),
            "row {y} of the top is not the same row on every tab: {:#?}",
            tops.iter().map(|top| &top[y]).collect::<Vec<_>>()
        );
    }
}

/// The Service tab has no status frame, so its region starts directly under the frame rule.
///
/// Chris, 2026-09-14: the status frame *"should be optional, since we won't need it in the
/// Service tab"*. Asserted on the frame AND on the screen: the first says the option exists,
/// the second says the hub actually takes it.
#[test]
fn the_service_tab_has_no_status_frame_and_its_region_starts_under_the_rule() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let area = Rect::new(0, 0, WIDE, TALL);
    let (bare, region) = page(
        crate::widgets::chrome::app_bar::SERVICE_TAB,
        false,
        WIDE,
        TALL,
    );
    assert_eq!(
        region.y,
        area.y + CONSTANT_ROWS,
        "with no status frame the region opens on the row below the frame rule"
    );
    assert_eq!(region.height, TALL - Page::CHROME_ROWS);

    // And the same rows WITH a frame, so the difference is the frame and not the arithmetic.
    let (_, framed) = page(0, true, WIDE, TALL);
    assert_eq!(
        framed.y - region.y,
        status_block::ROWS_FULL,
        "the frame is what the Service tab is not spending those rows on"
    );

    // The hub itself: nothing anywhere on it says `Service status`, which is the frame's own
    // first word. Its band says the same thing in more detail, and two claims about one system
    // on one screen is how the two come to disagree.
    let hub = dump(WIDE, TALL, |area, buf| {
        crate::views::service::frames::base().render(area, buf)
    });
    assert!(
        !hub.contains("Service status"),
        "the Service hub is carrying the status frame as well as its band"
    );
    assert!(
        !bare.contains("Service status"),
        "the frame drew a status frame it was never given"
    );
}

/// Every screen ends in a hairline and one help row — the bottom half of Chris's 2026-09-14
/// sentence, on all four.
#[test]
fn every_screen_ends_in_a_hairline_and_one_help_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let screens: [(&str, String); 4] = [
        (
            "bare page",
            dump(WIDE, TALL, |area, buf| {
                crate::views::shell::frames::dashboard().render(area, buf)
            }),
        ),
        (
            "dashboard",
            dump(WIDE, TALL, |area, buf| {
                crate::views::dashboard::ingredient::populated().render(area, buf)
            }),
        ),
        (
            "queue",
            dump(WIDE, TALL, |area, buf| {
                crate::views::queue::ingredient::queue(Default::default()).render(area, buf)
            }),
        ),
        (
            "service",
            dump(WIDE, TALL, |area, buf| {
                crate::views::service::frames::base().render(area, buf)
            }),
        ),
    ];

    let rule = RULE.repeat(WIDE as usize);
    for (name, text) in &screens {
        let lines: Vec<&str> = text.lines().collect();
        let hairline = TALL as usize - FOOT_ROWS as usize;
        assert_eq!(
            lines[hairline], rule,
            "{name} does not close its content with an edge-to-edge hairline"
        );
        assert!(
            !lines[hairline + 1].trim().is_empty(),
            "{name} has a hairline over an empty help row"
        );
        assert!(
            !lines[hairline - 1].contains(RULE),
            "{name} has two rules in a row — the one above the foot is content"
        );
    }
}

/// The two hints that always survive do survive a foot with no room for the rest.
///
/// Chris, 2026-09-07: when the hints do not fit, render `? Help · q Quit` alone, *"the help
/// modal carries them all"*. The rule lives in [`crate::widgets::chrome::status_line`]; what
/// this checks is that the page's foot actually goes through it, at a width where the
/// difference shows.
#[test]
fn the_pair_that_always_survives_survives_a_narrow_foot() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let wordy = |width: u16| -> String {
        dump(width, 12, |area, buf| {
            let mut page = Page::new(0);
            for (key, label) in [
                ("\u{2193}\u{2191}/jk", "Navigate"),
                ("/", "Search"),
                ("f", "Filter"),
                ("y", "Retry"),
                ("c", "Cancel"),
                ("x", "Remove"),
            ] {
                page = page.hint(key, label);
            }
            for (key, label) in ALWAYS {
                page = page.hint(key, label);
            }
            page.draw(area, buf);
        })
    };

    let roomy = wordy(WIDE);
    let foot = |text: &str| -> String {
        text.lines()
            .nth(12 - FOOT_ROWS as usize + 1)
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    assert!(
        foot(&roomy).contains("Retry"),
        "at 125 columns every hint fits — otherwise the narrow case below proves nothing: {:?}",
        foot(&roomy)
    );

    let narrow = wordy(80);
    assert_eq!(
        foot(&narrow),
        "? Help   q Quit",
        "an 80-column foot keeps exactly the pair that always survives"
    );
}

/// A frame with nowhere to put a view draws nothing, and says so by returning an empty region.
///
/// Half a screen is a rendering artefact rather than a screen: a chrome drawn over an area
/// that cannot hold a view is two decorations with nothing between them, which reads as a
/// broken app rather than as a small terminal.
#[test]
fn an_area_too_small_for_the_chrome_gets_an_empty_region_and_no_paint() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    for height in 0..=Page::CHROME_ROWS {
        let mut buf = Buffer::empty(Rect::new(0, 0, WIDE, height));
        let region = Page::new(0)
            .status(block())
            .hint("?", "Help")
            .draw(buf.area, &mut buf);
        assert_eq!(
            region.height, 0,
            "a {height}-row area got a region to draw in"
        );
        assert!(
            buffer_rows(&buf).trim().is_empty(),
            "a {height}-row area was painted: {:?}",
            buffer_rows(&buf)
        );
    }

    // One row more and there is a page: the guard has to have an edge, or it is a guard
    // against every height.
    let (_, region) = page(0, true, WIDE, Page::CHROME_ROWS + 1);
    assert_eq!(region.height, 1, "one row of view is a page");
}

/// Under a modal, not one cell of any page-drawn screen carries a colour.
///
/// VL §6, Chris 2026-09-07: *"we still have colors on the screen while all should be muted
/// (including the indicators)"*. Each screen had grown its own sweep; the rule is about every
/// page, so it is asked of every page in one place. The per-screen guards stay — they name the
/// cells each screen was caught painting, which this cannot.
#[test]
fn no_page_drawn_screen_carries_a_colour_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let neutrals = neutral_rungs();
    let render = |draw: &dyn Fn(Rect, &mut Buffer)| -> Buffer {
        let area = Rect::new(0, 0, WIDE, TALL);
        let mut buf = Buffer::empty(area);
        draw(area, &mut buf);
        buf
    };

    /// A screen, and the two ways it is drawn: live, and beneath a modal.
    type Case<'a> = (
        &'a str,
        &'a dyn Fn(Rect, &mut Buffer),
        &'a dyn Fn(Rect, &mut Buffer),
    );

    let cases: [Case; 3] = [
        (
            "bare page",
            &|area, buf| crate::views::shell::frames::dashboard().render(area, buf),
            &|area, buf| {
                crate::views::shell::frames::dashboard()
                    .under_modal(true)
                    .render(area, buf)
            },
        ),
        (
            "dashboard",
            &|area, buf| crate::views::dashboard::ingredient::populated().render(area, buf),
            &|area, buf| {
                crate::views::dashboard::ingredient::populated()
                    .under_modal(true)
                    .render(area, buf)
            },
        ),
        (
            "queue",
            &|area, buf| {
                crate::views::queue::ingredient::queue(Default::default()).render(area, buf)
            },
            &|area, buf| {
                crate::views::queue::ingredient::queue(Default::default())
                    .under_modal(true)
                    .render(area, buf)
            },
        ),
    ];

    for (name, live, under) in cases {
        // The live frame first: a screen that painted no colour anyway would pass the sweep
        // without the switch existing at all.
        assert!(
            !coloured_cells(&render(live), &neutrals).is_empty(),
            "{name} paints no colour even when it is live — this guard checks nothing"
        );
        let survivors = coloured_cells(&render(under), &neutrals);
        assert!(
            survivors.is_empty(),
            "{name} kept {} coloured cells under a modal, first ten: {:?}",
            survivors.len(),
            &survivors[..survivors.len().min(10)]
        );
    }

    // The Service hub is the fourth, and it is asked differently because it is the one screen
    // that draws a REAL modal: the box over it keeps its own colours by design, so the sweep
    // has to exclude the rectangle it covers. That guard is `views::service::tests::under_modal`,
    // which also pins where the scope closes — the assertion here would be a weaker copy of it.
    // What belongs here is only that the hub's page goes through the same switch.
    let hub = render(&|area, buf| crate::views::service::frames::confirming().render(area, buf));
    let rect = crate::views::service::frames::confirm_modal().rect(Rect::new(0, 0, WIDE, TALL));
    let outside: Vec<_> = coloured_cells(&hub, &neutrals)
        .into_iter()
        .filter(|(x, y, _, _)| {
            !((rect.x..rect.right()).contains(x) && (rect.y..rect.bottom()).contains(y))
        })
        .collect();
    assert!(
        outside.is_empty(),
        "the Service hub kept {} coloured cells behind its modal, first ten: {:?}",
        outside.len(),
        &outside[..outside.len().min(10)]
    );
}
