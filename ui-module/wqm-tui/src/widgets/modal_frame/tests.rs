//! One test per claim the Container and the Decoration make about geometry.
//!
//! Everything here renders into a [`Buffer`] and reads cells back, which is this crate's own
//! idiom: a claim about where a row lands is only settled by looking at where it landed.

use super::*;
use crate::encoding::Encoding;
use crate::tokens::{ModalTint, Palette};

/// The storyboard's own screen, and the floor Chris named.
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

struct Restore(Palette, Encoding, ModalTint, f32);

impl Restore {
    fn mocha() -> Self {
        let restore = Restore(
            Palette::current(),
            Encoding::current(),
            ModalTint::current(),
            tokens::tint_strength(),
        );
        Palette::set(Palette::Bundled);
        Encoding::set(Encoding::TrueColor);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        restore
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        Palette::set(self.0);
        Encoding::set(self.1);
        ModalTint::set(self.2);
        tokens::set_tint_strength(self.3);
    }
}

fn deco() -> Decoration {
    Decoration::new("open-books — queue")
        .crumbs(vec!["Libraries", "open-books", "Queue"])
        .hint("↓↑/jk", "Move")
        .hint("↵", "Drill down")
        .hint("⌫", "Back")
        .hint("?", "Help")
        .hint("q", "Close")
}

fn draw(area: Rect, container: Container) -> Buffer {
    let mut buf = Buffer::empty(area);
    let rect = Container::footprint(area);
    container.render(rect, &mut buf);
    buf
}

fn text(buf: &Buffer, row: u16) -> String {
    (buf.area.x..buf.area.right())
        .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
        .collect()
}

/// One row of the window's INSIDE — between the borders, padding included.
///
/// Reading the whole screen row instead puts a `│` at each end of every assertion, which is
/// how the first cut of these tests managed to claim a blank row was not blank.
fn inside(buf: &Buffer, rect: Rect, row: u16) -> String {
    (rect.x + 1..rect.right() - 1)
        .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
        .collect()
}

/// Where the two help rows land: the last two rows inside the border, under one blank.
fn help_rows(rect: Rect) -> (u16, u16) {
    (rect.bottom() - 3, rect.bottom() - 2)
}

/// [`Container::footprint`] is the ruled maximum and nothing else, at both sizes a frame is
/// drawn at - the page inset by five on every side.
///
/// Round 1's two arms were measured here and are gone: the 19:05 ruling dissolved the question
/// they answered rather than picking one, so there is no longer a pair to compare.
#[test]
fn the_default_footprint_is_the_page_inset_by_five() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    for area in [SCREEN, FLOOR] {
        let rect = Container::footprint(area);
        assert_eq!(
            (rect.width, rect.height),
            (area.width - 10, area.height - 10),
            "at {}x{}",
            area.width,
            area.height
        );
        assert_eq!((rect.x, rect.y), (area.x + 5, area.y + 5));
    }
}

/// The ruling's cost, pinned so it reads as a decision and not as a regression: at 125x34 the
/// top border lands on row 5, inside the three-row status block. Round 1 clamped windows below
/// the page header to avoid exactly this; item 0 says five rows from the screen, and it was
/// ruled after Chris had seen the clamped placement.
#[test]
fn the_literal_inset_puts_the_top_border_inside_the_status_block() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    assert_eq!(rect.y, SCREEN.y + 5, "five rows down, as worded");
    assert!(
        rect.y < SCREEN.y + crate::views::page::CONSTANT_ROWS + 4,
        "and that is above where the page's own header ends - the cost the ruling accepts"
    );
}

/// A fixed container is one whose size does not depend on what is inside it. Stated as: two
/// decorations that differ in every way a decoration can differ get the same rect.
#[test]
fn the_window_does_not_resize_between_views() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let bare = Container::new(Decoration::new("x"));
    let full = Container::new(deco().search(SearchRow::filter("open-books", false)))
        .scroll(Scroll {
            offset: 40,
            total: 900,
        })
        .title_banner("-- EDIT --");
    // Nothing about either container reaches `footprint`, and that IS the property.
    let _ = (&bare, &full);
    assert_eq!(Container::footprint(SCREEN), Container::footprint(SCREEN));
    assert_eq!(bare.viewport(Container::footprint(SCREEN)).width, 111);
    assert_eq!(full.viewport(Container::footprint(SCREEN)).width, 110);
}

/// The fifth row costs a row of VIEW, never a row of window — which is the whole reason it is
/// declared at composition time rather than summoned by a keypress.
#[test]
fn the_optional_fifth_row_takes_from_the_viewport_and_not_from_the_window() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let without = Container::new(deco());
    let with = Container::new(deco().search(SearchRow::search("guide", true)));

    assert_eq!(without.decoration().top_rows(), TOP_ROWS);
    assert_eq!(with.decoration().top_rows(), TOP_ROWS + 1);
    assert_eq!(
        with.viewport(rect).height + 1,
        without.viewport(rect).height,
        "the fifth row must come out of the view"
    );
    assert_eq!(
        with.viewport(rect).y,
        without.viewport(rect).y + 1,
        "and the view must start one row lower"
    );
}

/// Present when declared, absent when not — and absent means the row is the view's, not a
/// blank one nobody uses.
#[test]
fn the_search_row_is_drawn_only_when_the_view_declares_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let with = draw(
        SCREEN,
        Container::new(deco().search(SearchRow::filter("open-books", false))),
    );
    let fifth = rect.y + 1 + TOP_ROWS;
    assert!(
        inside(&with, rect, fifth).contains("open-books"),
        "the filter term is not on the fifth row: {:?}",
        inside(&with, rect, fifth)
    );

    let without = draw(SCREEN, Container::new(deco()));
    assert!(
        inside(&without, rect, fifth).trim().is_empty(),
        "a view with no search must leave that row to its content: {:?}",
        inside(&without, rect, fifth)
    );
}

/// The trail is what the composition handed in, in that order — start-dependent, never derived
/// from the view. A container that worked out its own breadcrumb would be asserting a
/// hierarchy the navigation does not have.
#[test]
fn the_breadcrumb_is_the_path_the_reader_took() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);

    let one = draw(SCREEN, Container::new(deco()));
    let crumbs = inside(&one, rect, rect.y + 1);
    let trail = crumbs.trim();
    assert!(trail.starts_with("Libraries"), "{trail:?}");
    assert!(trail.contains(CHEVRON.trim()), "no chevron in {trail:?}");
    assert!(
        trail.find("Libraries") < trail.find("open-books")
            && trail.find("open-books") < trail.find("Queue"),
        "the crumbs are out of order: {trail:?}"
    );

    // The same view reached the other way round shows the other trail.
    let other = draw(
        SCREEN,
        Container::new(deco().crumbs(vec!["Queue", "open-books", "reading_guide.py"])),
    );
    assert!(inside(&other, rect, rect.y + 1).trim().starts_with("Queue"));
}

/// Two help rows are RESERVED, not grown into. One key or nine, the content region is the same
/// height — a help area that grew a row when a view gained a key would move the content under
/// the reader for a reason they cannot see.
#[test]
fn two_help_rows_are_reserved_even_when_one_is_blank() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let one_key = Container::new(Decoration::new("t").hint("q", "Close"));
    let many = Container::new(deco());
    assert_eq!(one_key.viewport(rect).height, many.viewport(rect).height);

    let buf = draw(SCREEN, one_key);
    let (first, second) = help_rows(rect);
    assert!(
        inside(&buf, rect, first).contains("Close"),
        "the first help row fills: {:?}",
        inside(&buf, rect, first)
    );
    assert!(
        inside(&buf, rect, second).trim().is_empty(),
        "the second is reserved and blank: {:?}",
        inside(&buf, rect, second)
    );
    // …and the reserved row is INSIDE the window, one row above its bottom edge, rather than
    // a row the window never had. Read off the FILL: item (e) keeps the border's room and
    // drops its glyphs, so the left edge is a painted cell rather than a `│`.
    assert_eq!(
        buf.cell((rect.x, second)).expect("the window's left edge").bg,
        tokens::modal_fill(tokens::layer1_bg())
    );
    assert_eq!(
        second + 1,
        rect.bottom() - 1,
        "the bottom border follows it"
    );
}

/// More hints than one row holds spill onto the second, greedily — the first fills first, and
/// nothing lands half-drawn.
#[test]
fn the_help_packer_fills_the_first_row_before_the_second() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let mut deco = Decoration::new("t");
    for n in 0..6 {
        deco = deco.hint(format!("k{n}"), format!("a rather long label {n}"));
    }
    let buf = draw(SCREEN, Container::new(deco));
    let (top, bottom) = help_rows(rect);
    let first = inside(&buf, rect, top);
    let second = inside(&buf, rect, bottom);

    assert!(first.contains("label 0"), "the first row leads: {first:?}");
    assert!(
        !second.trim().is_empty(),
        "the overflow must land on the second row: {second:?}"
    );
    assert!(
        second.contains("label 5"),
        "the last hint is missing: {second:?}"
    );
    // Every label that appears appears whole. The defect this catches is a key named `k6 a
    // rather`, which is worse than a key absent: a truncated hint tells the reader a chord
    // that does not exist.
    for n in 0..6 {
        let label = format!("a rather long label {n}");
        let drawn = first.contains(&label) || second.contains(&label);
        let started = first.contains(&format!("k{n} ")) || second.contains(&format!("k{n} "));
        assert_eq!(drawn, started, "hint k{n} is drawn but its label is cut");
    }
}

/// A view that declares more hints than two rows hold loses the surplus WHOLE rather than
/// mid-word, and says so loudly in a test build. `?` is where the complete list lives.
#[test]
fn hints_past_the_two_row_budget_are_dropped_rather_than_cut() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let mut deco = Decoration::new("t");
    for n in 0..12 {
        deco = deco.hint(format!("k{n}"), format!("a rather long label {n}"));
    }
    // The `debug_assert` in the packer fires on the over-declaration, which is the signal a
    // frame needs; catching it here is what makes that signal a test rather than a surprise.
    let over = std::panic::catch_unwind(move || {
        let mut buf = Buffer::empty(SCREEN);
        Container::new(deco).render(rect, &mut buf);
    });
    assert!(
        over.is_err(),
        "twelve long hints must not fit two rows — if they now do, the budget moved"
    );
}

/// A bar appears only when there is travel to describe, its thumb never vanishes, and it moves
/// with the offset. Three claims, because a bar that fails any one of them is decoration.
#[test]
fn the_scrollbar_describes_the_travel_and_nothing_else() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let bar_column = |scroll: Scroll| {
        let container = Container::new(deco()).scroll(scroll);
        let viewport = container.viewport(rect);
        let buf = draw(SCREEN, Container::new(deco()).scroll(scroll));
        let x = viewport.x + viewport.width;
        (0..viewport.height)
            .map(|row| {
                buf.cell((x, viewport.y + row))
                    .expect("bar cell")
                    .symbol()
                    .to_string()
            })
            .collect::<Vec<_>>()
    };

    let fits = bar_column(Scroll {
        offset: 0,
        total: 3,
    });
    assert!(
        fits.iter().all(|glyph| glyph == " "),
        "content that fits gets no bar: {fits:?}"
    );

    let thumb_at = |offset: usize| {
        bar_column(Scroll { offset, total: 900 })
            .iter()
            .position(|glyph| glyph == "\u{2590}")
            .expect("a thumb, however short")
    };
    assert_eq!(thumb_at(0), 0, "at the top the thumb is at the top");
    assert!(thumb_at(450) > thumb_at(0), "the thumb must travel");
    assert!(thumb_at(900) > thumb_at(450), "…all the way down");
}

/// The window's fill is the crate's one blend, so a window and the fields inside it cannot
/// drift apart when the strength is retuned.
#[test]
fn the_window_is_filled_through_the_one_blend() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    ModalTint::set(ModalTint::Accent);
    for strength in [0.14f32, 0.28, 0.40] {
        tokens::set_tint_strength(strength);
        let rect = Container::footprint(SCREEN);
        let buf = draw(SCREEN, Container::new(deco()));
        assert_eq!(
            buf.cell((rect.x + 1, rect.y + 1)).expect("inside").bg,
            tokens::modal_fill(tokens::layer1_bg()),
            "at strength {strength}"
        );
        // The border's HUE is still the tint's, and the retired `Edge::Bordered` arm is
        // where that can be read: the default edge draws no glyph to carry a foreground.
        let bordered = draw(SCREEN, Container::new(deco()).edge(Edge::Bordered));
        assert_eq!(
            bordered.cell((rect.x, rect.y)).expect("border").fg,
            tokens::modal_border()
        );
    }
}

/// A window occludes. Stated against a buffer with something in it, because on an empty one an
/// opaque box and a transparent one are the same box.
#[test]
fn the_window_covers_the_page_rather_than_tinting_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut buf = Buffer::empty(SCREEN);
    for y in 0..SCREEN.height {
        for x in 0..SCREEN.width {
            buf.cell_mut((x, y)).expect("cell").set_symbol("x");
        }
    }
    let rect = Container::footprint(SCREEN);
    Container::new(deco()).render(rect, &mut buf);

    let blank = inside(&buf, rect, rect.y + 2);
    assert!(
        blank.trim().is_empty(),
        "the window's blank row shows the page through it: {blank:?}"
    );
    assert_eq!(
        buf.cell((rect.x - 1, rect.y + 2)).expect("cell").symbol(),
        "x",
        "…and it clears no more than it owns"
    );
}

/// A window too short for its own decoration draws nothing at all, rather than drawing the
/// help rows over the content or off the bottom edge.
#[test]
fn a_window_that_cannot_hold_its_decoration_draws_nothing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let tiny = Rect::new(0, 0, 40, TOP_ROWS + BOTTOM_ROWS + 1);
    let mut buf = Buffer::empty(tiny);
    Container::new(deco()).render(tiny, &mut buf);
    assert_eq!(buf, Buffer::empty(tiny));
}

/// The mode banner is words and weight, never hue — so it survives `NO_COLOR` and cannot be
/// read as a selection (r06 #8).
#[test]
fn the_mode_banner_is_bold_and_carries_no_hue() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    Encoding::set(Encoding::NoColor);
    let rect = Container::footprint(SCREEN);
    let buf = draw(SCREEN, Container::new(deco()).title_banner("-- EDIT --"));
    let title = text(&buf, rect.y + 3);
    assert!(title.contains("open-books"), "the title: {title:?}");
    assert!(title.contains("-- EDIT --"), "the banner: {title:?}");

    let at = title.find("-- EDIT --").expect("the banner") as u16;
    let cell = buf.cell((at, rect.y + 3)).expect("banner cell");
    assert!(cell.modifier.contains(Modifier::BOLD), "weight carries it");
    assert_eq!(cell.fg, tokens::normal(), "and no hue does");
}
