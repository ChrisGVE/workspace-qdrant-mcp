//! **What each round-2 frame claims, asserted on the buffer it actually paints.**
//!
//! A rendered frame is the deliverable, and a frame is exactly the kind of artifact that looks
//! right while being wrong — a glyph in the private-use area shows as a blank in a text dump, an
//! attribute shows as nothing at all in a PNG, and a padding computed on markup length lines up
//! until one cell is wide. So every claim the notes make about a frame is pinned here, against
//! the buffer, rather than left to whoever reads the picture.

use ratatui::{buffer::Buffer, layout::Rect, style::Modifier};

use super::*;
use crate::encoding::Encoding;
use crate::tokens::{contrast, Palette};
use crate::views::modal_framework::record::DropDown;
use crate::widgets::modal_frame::INSET;

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

struct Restore(Palette, Encoding, Option<ratatui_themes::ThemePalette>);

impl Restore {
    fn mocha() -> Self {
        let restore = Restore(Palette::current(), Encoding::current(), tokens::theme());
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
        if let Some(theme) = self.2 {
            tokens::set_theme(theme);
        }
    }
}

fn render(area: Rect, draw: impl FnOnce(Rect, &mut Buffer)) -> Buffer {
    let mut buf = Buffer::empty(area);
    draw(area, &mut buf);
    buf
}

fn row_text(buf: &Buffer, area: Rect, y: u16) -> String {
    (area.x..area.right())
        .filter_map(|x| buf.cell((x, y)).map(|c| c.symbol().to_string()))
        .collect()
}

fn whole(buf: &Buffer, area: Rect) -> String {
    (area.y..area.bottom())
        .map(|y| row_text(buf, area, y))
        .collect::<Vec<_>>()
        .join("\n")
}

/// **Item 0, end to end.** Five columns and five rows of page survive at every size — measured
/// on the painted buffer, not on the rect the footprint returned.
///
/// The distinction matters: a rect can be right while the widget draws outside it, and the page
/// under the window is a real drawing that would show the overrun.
#[test]
fn the_window_leaves_five_of_page_on_every_side_at_every_size() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    for area in [SCREEN, FLOOR, WIDE] {
        let buf = render(area, |area, buf| {
            proposed(|| {
                Frame::default().draw(area, buf);
            });
        });
        let painted = whole(&buf, area);
        let rows: Vec<&str> = painted.split('\n').collect();

        // The top border is the first row carrying the window's corner.
        let top = rows
            .iter()
            .position(|r| r.contains('\u{250c}'))
            .unwrap_or_else(|| panic!("{area:?}: no window drawn"));
        let bottom = rows
            .iter()
            .rposition(|r| r.contains('\u{2514}'))
            .expect("a bottom border");
        assert_eq!(top, INSET as usize, "{area:?}: rows of page above");
        assert_eq!(
            rows.len() - 1 - bottom,
            INSET as usize,
            "{area:?}: rows of page below"
        );

        // **Character positions, not byte offsets.** `str::find` answers in bytes, and the
        // page's rule is made of `─` at three bytes each — so the first version of this read the
        // left margin as 15 on a frame whose margin is 5, and would have gone on reading it as
        // 15 however wrong the frame got. This is the measurement trap the design's own render
        // rules name, hit inside the instrument that checks for it.
        let column_of = |glyph: char| {
            rows[top]
                .chars()
                .position(|c| c == glyph)
                .unwrap_or_else(|| panic!("{area:?}: no {glyph} on the border row"))
        };
        let left = column_of('\u{250c}');
        let right = column_of('\u{2510}');
        assert_eq!(left, INSET as usize, "{area:?}: columns of page left");
        assert_eq!(
            rows[top].chars().count() - 1 - right,
            INSET as usize,
            "{area:?}: columns of page right"
        );
    }
}

/// **Item 0's floor.** A page under the threshold draws the message and no window.
#[test]
fn a_page_below_the_floor_draws_the_message_instead_of_a_window() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let cramped = Rect {
        x: 0,
        y: 0,
        width: 40,
        height: 18,
    };
    let buf = render(cramped, |area, buf| {
        proposed(|| {
            Frame::default().draw(area, buf);
        });
    });
    let painted = whole(&buf, cramped);
    assert!(
        !painted.contains('\u{250c}'),
        "no window may be drawn below the floor:\n{painted}"
    );
    assert!(
        painted.contains("Screen too small") || painted.contains("larger screen"),
        "the message must be there instead:\n{painted}"
    );
}

/// **Item 3, the underline.** Under truecolor an editable value cell carries NO underline; under
/// `NO_COLOR` it does.
///
/// Both halves in one test, because the claim is a CONTRAST between two encodings and half of it
/// proves nothing: a frame with no underline anywhere would pass the first assertion while
/// having lost the mark altogether.
#[test]
fn the_set_mark_drops_its_underline_only_where_colour_can_carry_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    // **Scoped to one editable field's value cell**, and the scoping is the point. A first cut
    // counted underlines over the whole screen and found 27 under truecolor — every one of them
    // a table HEADER, which this crate underlines end to end by ruling: the Queue page beneath
    // the window has one, and so does the reference column inside it. Counting the screen
    // measured a different design decision and called it this one.
    let underlines_at = |encoding: Encoding| {
        Encoding::set(encoding);
        let mut buf = Buffer::empty(SCREEN);
        let window = proposed(|| {
            Frame {
                mode: Mode::Edit {
                    at: 6,
                    edit: Some(Edit::insert("256")),
                },
                ..Frame::default()
            }
            .draw(SCREEN, &mut buf)
        });
        let (_, viewport) = window.expect("125x34 holds a window");
        // `Operation` is field 4 and editable — a SET cell, not the POINT one, which is the
        // mark being asked about.
        let row = (viewport.y..viewport.bottom())
            .find(|y| row_text(&buf, viewport, *y).contains("Operation"))
            .expect("the Operation row is on screen");
        let value_x = viewport.x
            + crate::views::modal_framework::record::GUTTER as u16
            + crate::views::modal_framework::record::W_LABEL as u16;
        (value_x..value_x + 20)
            .filter(|x| {
                buf.cell((*x, row))
                    .is_some_and(|c| c.modifier.contains(Modifier::UNDERLINED))
            })
            .count()
    };

    let rgb = underlines_at(Encoding::TrueColor);
    let none = underlines_at(Encoding::NoColor);
    Encoding::set(Encoding::TrueColor);

    assert_eq!(
        rgb, 0,
        "item 3 removes the underline where the fill can carry the SET mark"
    );
    assert!(
        none > 0,
        "…and puts it back where the ladder collapses onto slots, or `which fields may I \
         change` is carried by colour alone (r06 #8)"
    );
}

/// **Item 3, black text.** The active field's foreground is the derived one, and it is legible
/// on the fill under it.
///
/// The second half is the whole point: *"so we increase the contrast"* is a claim about a ratio,
/// and a test that only checked the colour would pass on a black that cannot be read.
#[test]
fn the_active_field_carries_text_that_can_be_read_on_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    for name in ratatui_themes::ThemeName::all() {
        tokens::set_theme(name.palette());
        proposed(|| {
            let fill = field::active_bg();
            let text = field::active_fg();
            let ratio = contrast::contrast_ratio(text, fill);
            assert!(
                ratio >= contrast::BODY_FLOOR,
                "{}: the active field's text is {ratio:.1}:1 on its own fill",
                name.display_name()
            );
        });
    }
}

/// **Item (b).** The `-- EDIT --` banner is gone, and the frame is genuinely in edit mode — so
/// the absence is the ruling rather than a frame that forgot to enter it.
#[test]
fn the_edit_banner_is_gone_and_the_frame_is_still_editing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = render(SCREEN, |area, buf| {
        proposed(|| {
            Frame {
                mode: Mode::Edit {
                    at: 6,
                    edit: Some(Edit::insert("256")),
                },
                ..Frame::default()
            }
            .draw(area, buf);
        });
    });
    let painted = whole(&buf, SCREEN);
    assert!(
        !painted.contains("EDIT"),
        "item (b): the columnar change is the indication\n{painted}"
    );
    assert!(
        painted.contains('\u{25b8}'),
        "…but the active-row mark must still be there, or this frame is not in edit mode at \
         all and proves nothing"
    );
}

/// **Item (e).** The borderless arm draws no box, and puts its content on exactly the columns
/// the bordered arm does.
///
/// The second half is what makes the pair a fair comparison: if the content moved, Chris would
/// be judging two layouts rather than one layout with and without ink.
#[test]
fn the_borderless_window_keeps_every_column_the_bordered_one_uses() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    let frame_of = |edge: Edge| {
        render(SCREEN, |area, buf| {
            proposed(|| {
                Frame {
                    edge,
                    ..Frame::default()
                }
                .draw(area, buf);
            })
        })
    };
    let bordered = whole(&frame_of(Edge::Bordered), SCREEN);
    let spacing = whole(&frame_of(Edge::Spacing), SCREEN);

    assert!(
        bordered.contains('\u{250c}'),
        "the bordered arm draws a box"
    );
    assert!(
        !spacing.contains('\u{250c}') && !spacing.contains('\u{2502}'),
        "the spacing arm draws none:\n{spacing}"
    );

    // Strip the box glyphs from the bordered frame and the two must agree cell for cell, which
    // is "the content did not move" stated as an equality rather than as an impression.
    let stripped: String = bordered
        .chars()
        .map(|c| match c {
            '\u{250c}' | '\u{2510}' | '\u{2514}' | '\u{2518}' | '\u{2502}' => ' ',
            // The window's own horizontal runs, but NOT the page's rules, which both frames
            // draw identically outside the window.
            other => other,
        })
        .collect();
    for (row, (a, b)) in stripped.split('\n').zip(spacing.split('\n')).enumerate() {
        // Only the window's own rows: the top and bottom border rows differ by a run of `─`
        // that the spacing arm replaces with the window fill.
        if a.contains('\u{2500}') || b.contains('\u{2500}') {
            continue;
        }
        assert_eq!(a, b, "row {row} moved between the two arms");
    }
}

/// **Item 1's second half.** The drill-down frame really does draw both bars.
///
/// Asserted by glyph, because that is the claim: the horizontal bar is the vertical bar's own
/// pair rotated, and a test that only checked "something was drawn on the last row" would pass
/// on a rule.
#[test]
fn the_wide_table_frame_draws_a_vertical_and_a_horizontal_bar() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut buf = Buffer::empty(SCREEN);
    let window = proposed(|| wide_table(SCREEN, &mut buf)).expect("125x34 holds a window");
    // **Scoped to the window**, because the page beneath it draws scrollbars of its own with the
    // same glyphs — a whole-screen search would have passed on the Queue tab's bar while this
    // window had none, which is how the first version of this test passed on a frame whose
    // vertical bar was genuinely missing.
    let painted = whole(&buf, window);
    assert!(
        painted.contains('\u{2590}'),
        "the vertical bar's thumb is missing:\n{painted}"
    );
    assert!(
        painted.contains('\u{2584}'),
        "the horizontal bar's thumb is missing:\n{painted}"
    );
}

/// **Item (a).** The trail is drawn with powerline separators, and there is one for each
/// transition the item describes.
///
/// The count is the test worth having: the glyph lives in the private-use area, so it renders as
/// a blank in a text dump and as a replacement box on a terminal without a patched font — a
/// human reading the frame cannot count them and this can.
#[test]
fn the_trail_draws_one_separator_per_transition() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    // depth → separators: one between each pair of ancestors, one onto the current crumb, and
    // one closing it onto the window.
    for (crumbs, expected) in [
        (&["Configuration"][..], 2usize),
        (&["Queue", "open-books"][..], 2),
        (&["Queue", "open-books", "reading_guide.py"][..], 3),
    ] {
        let leaked: Vec<&'static str> = crumbs
            .iter()
            .map(|s| Box::leak(s.to_string().into_boxed_str()) as &'static str)
            .collect();
        let crumbs: &'static [&'static str] = Box::leak(leaked.into_boxed_slice());
        let buf = render(SCREEN, |area, buf| {
            proposed(|| {
                Frame {
                    crumbs,
                    ..Frame::default()
                }
                .draw(area, buf);
            });
        });
        let painted = whole(&buf, SCREEN);
        let found = painted.matches('\u{e0b0}').count();
        assert_eq!(
            found,
            expected,
            "depth {} drew {found} separators, expected {expected}",
            crumbs.len()
        );
    }
}

/// **Item 3's list.** The frameless drop-down draws no box and opens on the current value.
#[test]
fn the_drop_down_has_no_frame_and_opens_on_the_current_value() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    let anchor = Rect {
        x: 20,
        y: 10,
        width: 30,
        height: 1,
    };
    let (choices, at) = crate::views::modal_framework::frames::chunking();
    let list = DropDown::new(choices.clone(), at, anchor).item_three();
    let rect = list.rect(SCREEN);
    let buf = render(SCREEN, |area, buf| {
        proposed(|| {
            ratatui::widgets::Widget::render(
                DropDown::new(choices.clone(), at, anchor).item_three(),
                area,
                buf,
            );
        })
    });

    let painted = whole(&buf, rect);
    assert!(
        !painted.contains('\u{250c}') && !painted.contains('\u{2502}'),
        "item 3 removes the frame:\n{painted}"
    );
    let first = row_text(&buf, rect, rect.y);
    assert!(
        first.contains(&choices[at]),
        "the current value must be the first row, got {first:?}"
    );
}

/// The fuzzy filter narrows by subsequence, which is what a reader means by fuzzy.
#[test]
fn the_typed_filter_narrows_the_list_by_subsequence() {
    let (choices, at) = crate::views::modal_framework::frames::chunking();
    let anchor = Rect {
        x: 20,
        y: 10,
        width: 30,
        height: 1,
    };
    let all = DropDown::new(choices.clone(), at, anchor).item_three();
    let narrowed = DropDown::new(choices.clone(), at, anchor).filter("tsf");

    assert!(narrowed.visible().len() < all.visible().len());
    assert!(
        narrowed
            .visible()
            .iter()
            .all(|c| c.contains("tree-sitter/function")),
        "`tsf` is a subsequence of `tree-sitter/function` and of nothing else here: {:?}",
        narrowed.visible()
    );
}

/// **Item (c).** The window under a confirm is quietened the way the page is — so its hues are
/// gone, and the guard's are not.
#[test]
fn a_confirm_quietens_the_window_beneath_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    let buf = render(SCREEN, |area, buf| {
        proposed(|| {
            confirm(area, buf);
        });
    });
    let guard_rect = crate::views::modal_framework::stack::discard_guard()
        .rect(Footprint::Max.window(SCREEN).expect("a window"));

    // The window's own region, outside the guard: nothing there may still be wearing the data
    // cursor's hue, which is the loudest thing a record view paints.
    let window = Footprint::Max.window(SCREEN).expect("a window");
    let cursor = tokens::cursor_bg();
    let loud = (window.y..window.bottom())
        .flat_map(|y| (window.x..window.right()).map(move |x| (x, y)))
        .filter(|(x, y)| {
            !(guard_rect.x..guard_rect.right()).contains(x)
                || !(guard_rect.y..guard_rect.bottom()).contains(y)
        })
        .filter(|(x, y)| buf.cell((*x, *y)).is_some_and(|c| c.bg == cursor))
        .count();
    assert_eq!(
        loud, 0,
        "the window beneath the guard still carries the cursor hue — item (c) asks for the \
         same treatment the page gets"
    );
}
