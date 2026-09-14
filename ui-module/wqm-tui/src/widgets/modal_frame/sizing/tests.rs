//! The size model, measured — item 0's numbers rather than anybody's reading of them.

use super::*;
use crate::widgets::modal_frame::{Container, Decoration};

/// The storyboard's screen, the floor Chris named, and a wide terminal.
const SCREEN: Rect = Rect {
    x: 0,
    y: 0,
    width: 125,
    height: 34,
};
const FLOOR: Rect = Rect {
    x: 0,
    y: 0,
    width: 100,
    height: 30,
};
const WIDE: Rect = Rect {
    x: 0,
    y: 0,
    width: 200,
    height: 40,
};

/// **Item 0, literally.** Five columns of page each side and five rows top and bottom, at every
/// size — which is what *"always relative to the background"* means.
///
/// Asserted as the leftover page rather than as a width, because the ruling is about what is
/// LEFT rather than about how big the window is: a width assertion would pass on a window that
/// was the right size in the wrong place.
#[test]
fn the_maximum_window_leaves_five_columns_and_five_rows_of_page() {
    for area in [SCREEN, FLOOR, WIDE] {
        let rect = Footprint::Max
            .window(area)
            .expect("all three hold a window");
        assert_eq!(rect.x - area.x, INSET, "{area:?}: left margin");
        assert_eq!(area.right() - rect.right(), INSET, "{area:?}: right margin");
        assert_eq!(rect.y - area.y, INSET, "{area:?}: top margin");
        assert_eq!(
            area.bottom() - rect.bottom(),
            INSET,
            "{area:?}: bottom margin"
        );
    }
}

/// The three sizes the notes quote, pinned so the table cannot go stale.
#[test]
fn the_maximum_window_at_the_three_sizes_the_notes_name() {
    let at = |area: Rect| {
        let r = Footprint::Max.window(area).expect("holds a window");
        (r.width, r.height)
    };
    assert_eq!(at(SCREEN), (115, 24));
    assert_eq!(at(FLOOR), (90, 20));
    assert_eq!(at(WIDE), (190, 30));
}

/// **The minimum, and it is derived.** The smallest window still draws its whole decoration and
/// exactly one row of content.
///
/// This is the measurement behind the number Chris asked for — *"I am not sure how small we can
/// accept a modal window to be"*. It is not a preference: at one row less the viewport is empty,
/// and a container that reports a zero-height viewport is a window with nothing in it.
#[test]
fn the_minimum_window_still_has_one_row_of_content() {
    let deco = Decoration::new("Title").hint("q", "Close");
    let container = Container::new(deco);

    let smallest = Rect {
        x: 0,
        y: 0,
        width: MIN_COLS,
        height: MIN_ROWS,
    };
    assert_eq!(
        container.viewport(smallest).height,
        1,
        "the minimum window must leave exactly one content row — if this is 0 the floor is too \
         low, if it is 2 the floor is higher than it needs to be"
    );

    let one_less = Rect {
        height: MIN_ROWS - 1,
        ..smallest
    };
    assert_eq!(
        Container::new(Decoration::new("Title").hint("q", "Close"))
            .viewport(one_less)
            .height,
        0,
        "one row below the floor there is no content row left, which is what makes it a floor"
    );
}

/// **The page floor follows the window floor**, so the too-small threshold and the minimum
/// window can never drift apart.
#[test]
fn a_page_one_cell_under_the_threshold_gets_no_window() {
    let exactly = Rect {
        x: 0,
        y: 0,
        width: MIN_PAGE_COLS,
        height: MIN_PAGE_ROWS,
    };
    assert!(fits(exactly), "the threshold itself must hold a window");
    let window = Footprint::Max.window(exactly).expect("at the threshold");
    assert_eq!((window.width, window.height), (MIN_COLS, MIN_ROWS));

    for short in [
        Rect {
            width: MIN_PAGE_COLS - 1,
            ..exactly
        },
        Rect {
            height: MIN_PAGE_ROWS - 1,
            ..exactly
        },
    ] {
        assert!(!fits(short), "{short:?} is under the threshold");
        assert_eq!(
            Footprint::Max.window(short),
            None,
            "{short:?} must refuse rather than return a window that cannot draw"
        );
    }
}

/// **Item 1's first half.** A content-sized window is as small as its content and no smaller
/// than the floor, and it never grows past the maximum.
#[test]
fn a_content_sized_window_is_capped_by_the_maximum_and_floored_by_the_minimum() {
    let (max_cols, max_rows) = Footprint::max_size(SCREEN);

    let modest = Footprint::Content { cols: 60, rows: 14 }
        .window(SCREEN)
        .expect("holds a window");
    assert_eq!((modest.width, modest.height), (60, 14));

    let greedy = Footprint::Content {
        cols: 400,
        rows: 400,
    }
    .window(SCREEN)
    .expect("holds a window");
    assert_eq!(
        (greedy.width, greedy.height),
        (max_cols, max_rows),
        "a view asking for more than the page has gets the maximum, not the page"
    );

    let tiny = Footprint::Content { cols: 4, rows: 2 }
        .window(SCREEN)
        .expect("holds a window");
    assert_eq!(
        (tiny.width, tiny.height),
        (MIN_COLS, MIN_ROWS),
        "a view asking for less than the floor gets the floor, not a window that cannot draw"
    );
}

/// A content-sized window is centred like every other one, so the two sizing policies do not
/// also become two placements.
#[test]
fn a_content_sized_window_sits_where_a_maximum_one_would() {
    let content = Footprint::Content { cols: 60, rows: 14 }
        .window(SCREEN)
        .expect("holds a window");
    let expected_x = SCREEN.x + (SCREEN.width - 60) / 2;
    let expected_y = SCREEN.y + (SCREEN.height - 14) / 2;
    assert_eq!((content.x, content.y), (expected_x, expected_y));
}

/// The message names both numbers, because a reader who is told only that the screen is too
/// small cannot tell how much to drag.
#[test]
fn the_too_small_message_says_what_is_needed_and_what_is_there() {
    let cramped = Rect {
        x: 0,
        y: 0,
        width: 30,
        height: 12,
    };
    let roomy = TooSmall::new(cramped)
        .detail(80)
        .expect("80 columns fits the full form");
    assert!(
        roomy.contains(&format!("{MIN_PAGE_COLS}x{MIN_PAGE_ROWS}")),
        "the message must name the size needed: {roomy}"
    );
    assert!(
        roomy.contains("30x12"),
        "the message must name the size present: {roomy}"
    );
}

/// **The message about a small screen has to fit a small screen**, and every form it can take
/// is checked rather than only the one a comfortable fixture produces.
///
/// The defect this pins was found by rendering, not by reading: at 30 columns the full headline
/// came out as `This window needs a larger scr`, cut mid-word, and a truncated sentence on an
/// otherwise empty screen is indistinguishable from a crash.
#[test]
fn every_form_of_the_too_small_message_fits_the_width_it_is_chosen_for() {
    let message = TooSmall::new(Rect {
        x: 0,
        y: 0,
        width: 30,
        height: 12,
    });
    // Down to a width no real terminal goes below, so the sweep covers the whole range rather
    // than the two widths that happened to be tried.
    for width in 8u16..=120 {
        let headline = message.headline(width);
        if width >= 33 {
            assert!(
                headline.chars().count() <= width as usize,
                "at {width} columns the headline {headline:?} does not fit"
            );
        }
        if let Some(detail) = message.detail(width) {
            assert!(
                detail.chars().count() <= width as usize,
                "at {width} columns the detail {detail:?} does not fit"
            );
        }
    }
    // The short headline is the floor: below its own width there is nothing shorter to fall
    // back to, and that is a stated limit rather than an oversight.
    assert_eq!(message.headline(8), "Screen too small");
    assert_eq!(message.detail(8), None, "no detail form fits 8 columns");
}

/// The message draws on a page too small for a window — the case it exists for — and puts real
/// glyphs on the buffer rather than returning early.
#[test]
fn the_too_small_message_draws_on_a_page_that_cannot_hold_a_window() {
    let cramped = Rect {
        x: 0,
        y: 0,
        width: 30,
        height: 12,
    };
    assert!(!fits(cramped), "the fixture must be a page that refuses");
    let mut buf = ratatui::buffer::Buffer::empty(cramped);
    TooSmall::new(cramped).render(cramped, &mut buf);

    let painted: String = (0..cramped.height)
        .flat_map(|y| (0..cramped.width).map(move |x| (x, y)))
        .filter_map(|(x, y)| buf.cell((x, y)).map(|c| c.symbol().to_string()))
        .collect();
    assert!(
        painted.contains("Screen too small"),
        "the message did not reach the buffer: {painted:?}"
    );
    assert!(
        painted.contains(&format!("{MIN_PAGE_COLS}x{MIN_PAGE_ROWS} needed")),
        "the short detail did not reach the buffer: {painted:?}"
    );
}
