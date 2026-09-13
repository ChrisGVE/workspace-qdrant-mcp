//! The frames are the deliverable, so each one is guarded: if it cannot be produced, the
//! feature is not done.

use super::*;
use crate::encoding::Encoding;
use crate::tokens::Palette;

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

fn draw(area: Rect, f: impl FnOnce(Rect, &mut Buffer)) -> Buffer {
    let mut buf = Buffer::empty(area);
    f(area, &mut buf);
    buf
}

fn screen_text(buf: &Buffer) -> String {
    let mut out = String::new();
    for y in buf.area.y..buf.area.bottom() {
        for x in buf.area.x..buf.area.right() {
            if let Some(cell) = buf.cell((x, y)) {
                out.push_str(cell.symbol());
            }
        }
        out.push('\n');
    }
    out
}

/// The page is under every frame, and it is QUIET — judged over the page it covers is the
/// whole reason these are screens rather than windows on an empty buffer.
#[test]
fn every_frame_is_a_whole_screen_with_the_page_beneath_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = draw(SCREEN, |area, buf| {
        RecordFrame::default().draw(area, buf);
    });
    let all = screen_text(&buf);
    // The page's own header is still readable above the window (Nielsen #1).
    assert!(all.contains("daemon"), "the status block is gone:\n{all}");
    assert!(all.contains("Queue item"), "and the window is not drawn");

    // The page is under the window, so it went through the muting scope: no cell of the page
    // carries the selector's reserved hue while a window is up.
    let window = Container::footprint(SCREEN);
    for y in 0..window.y {
        for x in 0..SCREEN.width {
            let cell = buf.cell((x, y)).expect("cell");
            assert_ne!(
                cell.bg,
                tokens::cursor_bg(),
                "the page still wears a highlight at ({x}, {y})"
            );
        }
    }
}

/// Every field kind is in the fixture, so a frame is capable of being wrong about one.
#[test]
fn the_fixture_exercises_every_field_kind_the_ruling_names() {
    let fields = record();
    let kinds = |f: &dyn Fn(&Value) -> bool| fields.iter().any(|row| f(row.value()));
    assert!(kinds(&|v| matches!(v, Value::Number(_))), "number");
    assert!(kinds(&|v| matches!(v, Value::Text(_))), "single-line text");
    assert!(kinds(&|v| matches!(v, Value::Multi(_))), "a multi-line box");
    assert!(kinds(&|v| matches!(v, Value::Bool(_))), "a tick box");
    assert!(kinds(&|v| matches!(v, Value::Radio { .. })), "a radio");
    assert!(kinds(&|v| matches!(v, Value::Choice { .. })), "a drop-down");
    assert!(
        fields.iter().any(|row| !row.is_editable()),
        "and a read-only field, or EDIT mode shows a wall of slots rather than a set"
    );
}

/// Every frame the round is judged from renders at 125×34, and again at the 100×30 floor. The
/// storyboard is the product: a frame that cannot be produced is a feature that is not done.
#[test]
fn every_frame_renders_at_both_sizes() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    type Frame = (&'static str, fn(Rect, &mut Buffer));
    let frames: Vec<Frame> = vec![
        ("record, view mode", |area, buf| {
            RecordFrame::default().draw(area, buf);
        }),
        ("record, edit mode", |area, buf| {
            RecordFrame::view(Mode::Edit {
                at: 6,
                edit: Some(crate::widgets::edit_field::Edit::insert("256")),
            })
            .draw(area, buf);
        }),
        ("record, two columns", |area, buf| {
            RecordFrame {
                reference: Reference::None,
                ..RecordFrame::default()
            }
            .draw(area, buf);
        }),
        ("record, empty", |area, buf| {
            RecordFrame {
                empty: true,
                ..RecordFrame::default()
            }
            .draw(area, buf);
        }),
        ("record, scrolled", |area, buf| {
            RecordFrame {
                offset: 5,
                ..RecordFrame::default()
            }
            .draw(area, buf);
        }),
        ("table in a modal", |area, buf| {
            table_frame(area, buf, false);
        }),
        ("table, pinned column shown", |area, buf| {
            table_frame(area, buf, true);
        }),
        ("table, empty", |area, buf| {
            empty_table_frame(area, buf);
        }),
        ("confirm over the window", |area, buf| {
            confirm_frame(area, buf);
        }),
        ("drop-down open", |area, buf| {
            dropdown_frame(area, buf);
        }),
        ("contextual help", |area, buf| {
            help_frame(area, buf);
        }),
        ("slide t=0", |area, buf| slide_frame(area, buf, 0.0)),
        ("slide t=0.5", |area, buf| slide_frame(area, buf, 0.5)),
        ("slide t=1", |area, buf| slide_frame(area, buf, 1.0)),
    ];

    for (name, render) in frames {
        for area in [SCREEN, FLOOR] {
            let buf = draw(area, render);
            let all = screen_text(&buf);
            assert!(
                all.chars().any(|c| !c.is_whitespace()),
                "{name} at {}×{} produced an empty frame",
                area.width,
                area.height
            );
            // The window's border is on screen, whole, at both sizes.
            let window = Container::footprint(area);
            assert!(
                window.bottom() <= area.bottom() && window.right() <= area.right(),
                "{name}: the window runs off a {}×{} screen",
                area.width,
                area.height
            );
        }
    }
}

/// The trail is at depth 3, which is what makes the chevrons a navigation aid rather than a
/// decoration on a one-item list — and the frame is the only place the depth is visible.
#[test]
fn the_record_frame_is_reached_at_depth_three() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    assert_eq!(
        RecordFrame::default().stack().crumbs(),
        CRUMBS.to_vec(),
        "the trail is not the one the fixture describes"
    );

    let buf = draw(SCREEN, |area, buf| {
        RecordFrame::default().draw(area, buf);
    });
    let rect = Container::footprint(SCREEN);
    let trail: String = (rect.x + 1..rect.right() - 1)
        .filter_map(|x| buf.cell((x, rect.y + 1)).map(|c| c.symbol().to_string()))
        .collect();
    assert_eq!(
        trail.matches('\u{203a}').count(),
        2,
        "three crumbs need two chevrons: {trail:?}"
    );
}

/// The picked radio button is a DIFFERENT GLYPH from its unpicked neighbours — read off the
/// rendered frame, because a PNG at 8×13 cells cannot settle it by eye and that is exactly the
/// sort of thing a reader would take on trust.
#[test]
fn the_picked_radio_button_is_filled_in_the_frame() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = draw(SCREEN, |area, buf| {
        RecordFrame::default().draw(area, buf);
    });
    let all = screen_text(&buf);
    assert!(
        all.contains("(\u{25cf}) update"),
        "the picked choice is not filled in the frame"
    );
    assert!(all.contains("( ) add"), "…and its neighbours are not");
    assert!(all.contains("( ) scan"), "…including the last one, whole");
}

/// The two footprint arms are both renderable, because Chris may overturn the gate's reading.
#[test]
fn both_footprint_arms_render() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    for footprint in [Footprint::Framework, Footprint::HelpDerived] {
        let buf = draw(SCREEN, |area, buf| {
            RecordFrame {
                footprint,
                ..RecordFrame::default()
            }
            .draw(area, buf);
        });
        assert!(screen_text(&buf).contains("Queue item"), "{footprint:?}");
    }
}

/// The tint bracket, rendered end to end, so the range is bracketed rather than guessed.
#[test]
fn the_tint_bracket_renders_at_every_strength() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let fill_at = |tint, strength| {
        with_tint(tint, strength, || {
            let buf = draw(SCREEN, |area, buf| {
                RecordFrame::default().draw(area, buf);
            });
            let rect = Container::footprint(SCREEN);
            buf.cell((rect.x + 1, rect.y + 1)).expect("inside").bg
        })
    };
    let neutral = fill_at(ModalTint::Neutral, PROPOSED_WASH);
    let quiet = fill_at(ModalTint::Accent, 0.14);
    let proposed = fill_at(ModalTint::Accent, PROPOSED_WASH);
    let loud = fill_at(ModalTint::Accent, 0.40);
    assert_eq!(neutral, tokens::layer1_bg(), "neutral blends nothing");
    for pair in [(neutral, quiet), (quiet, proposed), (proposed, loud)] {
        assert_ne!(pair.0, pair.1, "two arms of the bracket are one frame");
    }
}

/// The underline arm and the plain arm are different frames, and the difference is the SET
/// mark — which is the whole thing the pair exists to show.
#[test]
fn the_two_field_background_arms_differ_by_the_underline() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mark_at = |underlined| {
        let buf = draw(SCREEN, |area, buf| {
            RecordFrame {
                mode: Mode::Edit { at: 6, edit: None },
                underlined,
                ..RecordFrame::default()
            }
            .draw(area, buf);
        });
        let viewport = Container::new(RecordFrame::default().stack().decoration())
            .viewport(Container::footprint(SCREEN));
        // Field 7 (`Watch for changes`) is in the editable SET but is not the active field,
        // and it sits one row below the reference header.
        let y = viewport.y + 1 + 7;
        let x = viewport.x + (super::super::record::GUTTER + W_LABEL) as u16;
        buf.cell((x, y))
            .expect("an editable field")
            .modifier
            .contains(tokens::field::EDITABLE_MARK)
    };
    assert!(mark_at(true), "A+ rules its editable fields");
    assert!(!mark_at(false), "A does not — which is the objection to A");
}

/// Every frame survives the encodings Chris may be reading them in, and the SET mark survives
/// with them — which is why the underline is in.
#[test]
fn the_frames_survive_every_encoding() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    for encoding in [
        Encoding::TrueColor,
        Encoding::Ansi256,
        Encoding::Ansi16,
        Encoding::NoColor,
    ] {
        Encoding::set(encoding);
        let buf = draw(SCREEN, |area, buf| {
            RecordFrame::view(Mode::Edit { at: 6, edit: None }).draw(area, buf);
        });
        let all = screen_text(&buf);
        assert!(
            all.contains("-- EDIT --"),
            "{encoding:?}: the mode banner is gone, so the mode is carried by colour alone"
        );
        assert!(all.contains("Queue item"), "{encoding:?}: no window");
    }
}

/// The slide's middle frame really is between the two ends, or there is nothing to judge.
#[test]
fn the_slide_frames_are_three_different_frames() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let at = |t| screen_text(&draw(SCREEN, |area, buf| slide_frame(area, buf, t)));
    let (start, middle, end) = (at(0.0), at(0.5), at(1.0));
    assert_ne!(start, middle);
    assert_ne!(middle, end);
    assert_ne!(start, end);
}

/// The help window is a COMPOSITION — this container with the Queue's own help inside it — and
/// its content does not fit, which is the scroll the ruling asks for and the literal footprint
/// could never provide.
#[test]
fn the_help_window_is_a_container_whose_content_scrolls() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let lines = crate::panes::list::help::render(&Queue::help_sections());
    let rect = Container::footprint(SCREEN);
    let viewport = Container::new(crate::widgets::modal_frame::Decoration::new("x")).viewport(rect);
    assert!(
        lines.len() > viewport.height as usize,
        "the help content fits the window, so nothing here is scrolling: {} lines in {} rows",
        lines.len(),
        viewport.height
    );

    let buf = draw(SCREEN, |area, buf| {
        help_frame(area, buf);
    });
    let all = screen_text(&buf);
    assert!(all.contains("Navigation"), "the help sections are drawn");
    let bar_x = viewport.x + viewport.width - 1;
    let column: String = (viewport.y..viewport.y + viewport.height)
        .filter_map(|y| buf.cell((bar_x, y)).map(|c| c.symbol().to_string()))
        .collect();
    assert!(
        column.contains('\u{2590}'),
        "a scrollable help window must show its bar: {column:?}"
    );
}
