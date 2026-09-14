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

/// The four corners a window's border draws. A frame guard reads these back to decide that a
/// window was drawn at all.
const CORNERS: [&str; 4] = ["\u{250c}", "\u{2510}", "\u{2514}", "\u{2518}"];

/// **The detector, checked before anything is trusted to it.**
///
/// `every_frame_renders_at_both_sizes` concludes *a window was drawn* from a box corner in the
/// window's rect. That inference is only sound if the PAGE draws no corner of its own —
/// otherwise the guard passes on the fixture and reports nothing about the frame, which is
/// exactly the hollowness it was rewritten to escape. So the page is rendered alone and
/// required to produce none.
///
/// It is also a real invariant rather than a convenience: §6 gives a box exactly two meanings,
/// a modal and a toast, and Chris ruled *"no frame around the table, valid for all views"*. A
/// page that grew a border would be a design defect first and a broken detector second — and
/// this would fail on the design, which is the right order to find out.
///
/// # If this goes red, the fix is to the PAGE, not to this test
///
/// Said plainly because the obvious move on a red line here is to reach for a different
/// marker, and that is repairing the instrument until it agrees with the defect. A corner on
/// the page means the page has grown a box §6 does not allow it; the corner is the symptom and
/// the border is the bug. Changing the detector would leave the design broken and
/// [`every_frame_renders_at_both_sizes`] asserting nothing, which is where this pair started.
/// The next reader meets this failure without the conversation that produced it.
#[test]
fn a_window_corner_is_evidence_because_the_page_alone_draws_none() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    for area in [SCREEN, FLOOR] {
        let buf = draw(area, |area, buf| page().render(area, buf));
        for y in area.y..area.bottom() {
            for x in area.x..area.right() {
                let symbol = buf.cell((x, y)).expect("a cell on screen").symbol();
                assert!(
                    !CORNERS.contains(&symbol),
                    "the page drew {symbol:?} at ({x}, {y}) on a {}×{} screen, so a corner is \
                     no longer evidence that a window was drawn over it",
                    area.width,
                    area.height
                );
            }
        }
    }
}

/// Every frame the round is judged from renders at 125×34, and again at the 100×30 floor. The
/// storyboard is the product: a frame that cannot be produced is a feature that is not done.
///
/// What makes the corner check mean anything is
/// [`a_window_corner_is_evidence_because_the_page_alone_draws_none`], which establishes that
/// the fixture underneath cannot produce one.
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
            let window = Container::footprint(area);
            let at = |size: Rect| format!("{name} at {}×{}", size.width, size.height);

            // **The window was DRAWN**, all four corners of its border. The first version of
            // this asked only whether the screen held any non-whitespace character — which a
            // frame that drew one glyph would satisfy — and then checked that the footprint
            // RECT fitted the screen, which is arithmetic that holds whether or not anything
            // rendered at all. Neither assertion could tell a frame from an empty buffer.
            let corners = [
                (CORNERS[0], window.x, window.y),
                (CORNERS[1], window.right() - 1, window.y),
                (CORNERS[2], window.x, window.bottom() - 1),
                (CORNERS[3], window.right() - 1, window.bottom() - 1),
            ];
            for (corner, x, y) in corners {
                assert_eq!(
                    buf.cell((x, y)).expect("a cell on screen").symbol(),
                    corner,
                    "{}: no window corner at ({x}, {y})",
                    at(area)
                );
            }

            // …and the page is still readable ABOVE it, at both sizes. That is §6's depth
            // model stated as a consequence rather than as geometry: a window never costs the
            // reader the answer to *is anything wrong* (Nielsen #1), which is the whole
            // argument for the framework footprint over the literal one.
            let mut header = String::new();
            for row in 0..window.y {
                for x in 0..area.width {
                    if let Some(cell) = buf.cell((x, row)) {
                        header.push_str(cell.symbol());
                    }
                }
            }
            assert!(
                header.contains("daemon") && header.contains("pending"),
                "{}: the page's own status block is not readable above the window: {header:?}",
                at(area)
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
    let mark_at = |arm_a| {
        let buf = draw(SCREEN, |area, buf| {
            RecordFrame {
                mode: Mode::Edit { at: 6, edit: None },
                arm_a,
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
    // `mark_at` answers *is the underline there*, and `arm_a` names the REJECTED arm — so the
    // shipping frame is `arm_a: false` and it is the one that must carry the mark.
    assert!(mark_at(false), "the shipping arm rules its editable fields");
    assert!(
        !mark_at(true),
        "…and the rejected one does not, which is the objection to it"
    );
}

/// **The shipping arm is the only one a window can draw.** Arm A is reachable from the frame
/// generator, which is evidence, and from nowhere else — so it cannot be shipped by passing a
/// flag down from a setting.
///
/// Stated as: a record view built the ordinary way rules its editable fields, and nothing
/// short of the explicitly-named rejected-arm builder takes that away.
#[test]
fn a_window_cannot_be_asked_for_the_rejected_arm() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let stack = RecordFrame::view(Mode::Edit { at: 6, edit: None }).stack();
    let viewport = Container::new(stack.decoration()).viewport(rect);

    let mut buf = Buffer::empty(SCREEN);
    stack.render(rect, &mut buf);
    let y = viewport.y + 1 + 7;
    let x = viewport.x + (super::super::record::GUTTER + W_LABEL) as u16;
    assert!(
        buf.cell((x, y))
            .expect("an editable field")
            .modifier
            .contains(tokens::field::EDITABLE_MARK),
        "a stack renders the shipping arm and has no way to be told otherwise"
    );
}

/// **The adversarial frame really is adversarial** — verified off the rendered cells, not
/// inferred from the theme having been set.
///
/// `with_theme` has a way to fail silently that would leave the evidence looking fine: set the
/// theme without `Palette::Bundled` and `tokens::active_theme` answers `None`, so the hues fall
/// back to slots AND the neutral ladder falls back to the ambient endpoints — the frame would
/// then be Mocha's ladder wearing Solarized Dark's name. So this measures the two cells the
/// argument is about, straight out of the buffer, and checks they are as thin as the table
/// says.
#[test]
fn the_adversarial_theme_frame_shows_the_thin_ladder_it_claims_to() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let viewport = Container::new(RecordFrame::default().stack().decoration()).viewport(rect);
    // `Watch for changes` — in the editable SET, not the active field, one row below the
    // reference header.
    let field = (
        viewport.x + (super::super::record::GUTTER + W_LABEL) as u16,
        viewport.y + 1 + 7,
    );
    let window = (rect.x + 1, rect.y + 1);

    let lift_on = |theme| {
        with_theme(theme, || {
            let buf = draw(SCREEN, |area, buf| {
                RecordFrame::view(Mode::Edit { at: 6, edit: None }).draw(area, buf);
            });
            tokens::delta_e(
                buf.cell(window).expect("the window's fill").bg,
                buf.cell(field).expect("an editable field").bg,
            )
        })
    };

    let thin = lift_on(ADVERSARIAL_THEME);
    let roomy = lift_on(ratatui_themes::ThemeName::CatppuccinMocha);
    println!("SET lift in the rendered frame: {thin:.1} adversarial, {roomy:.1} Mocha");

    // The ground the whole frame sits on moved too, or the theme only reached the window and
    // the page around it is still somebody else's. Both the screen fill and the ladder's own
    // endpoints are checked, because they come from different accessors and either could be
    // the one that did not take.
    let ground = |theme| {
        with_theme(theme, || {
            (tokens::screen_bg(), tokens::ladder_endpoints().background)
        })
    };
    let (adversarial_bg, adversarial_end) = ground(ADVERSARIAL_THEME);
    let (mocha_bg, mocha_end) = ground(ratatui_themes::ThemeName::CatppuccinMocha);
    println!("ground: adversarial {adversarial_bg:?} / {adversarial_end:?}, mocha {mocha_bg:?} / {mocha_end:?}");
    assert_ne!(
        adversarial_bg, mocha_bg,
        "the page ground did not change theme"
    );
    assert_ne!(
        adversarial_end, mocha_end,
        "the ladder is still being built between the other theme's endpoints"
    );
    assert!(
        thin < roomy,
        "the adversarial theme must be the thinner one — {thin:.1} vs {roomy:.1}"
    );
    assert!(
        thin < 3.0,
        "the adversarial frame is showing a comfortable ladder (ΔE {thin:.1}), so either the \
         theme did not take or it has stopped being the worst case"
    );
}

/// **A themed WIDGET RENDER does not depend on the ambient terminal endpoints.**
///
/// Under a bundled theme every rung comes from `tokens::ladder_endpoints` — the theme's own
/// background and foreground — so `tokens::set_endpoints`, which is what `WQM_TUI_TERM_BG`/`FG`
/// reach, should change nothing a widget draws.
///
/// # This is half the claim, and the half it is not covers the other one
///
/// It renders straight into a [`Buffer`] and never goes through [`crate::capture`], so it says
/// nothing about the ground a PNG is painted on — that is
/// `capture::tests::a_themed_capture_paints_its_ground_from_the_theme_and_not_the_terminal`,
/// which captures an EMPTY grid for the purpose. Worth stating because the first version of
/// this test was described as pinning the whole byte-identity claim and did not: this frame is
/// a full-screen composition whose page paints `screen_bg()` edge to edge, so the ground has
/// nothing to show through and it would pass either way. Inert by coverage, not by
/// construction — the exact failure mode a test can wear as a green tick.
#[test]
fn a_themed_widget_render_ignores_the_terminals_own_endpoints() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let ambient = tokens::endpoints();

    let render_against = |background, foreground| {
        tokens::set_endpoints(crate::terminal::Endpoints {
            background,
            foreground,
        });
        draw(SCREEN, |area, buf| {
            RecordFrame::default().draw(area, buf);
        })
    };

    let black_on_white = render_against(
        crate::terminal::Rgb::new(0, 0, 0),
        crate::terminal::Rgb::new(0xff, 0xff, 0xff),
    );
    let mocha = render_against(
        crate::terminal::Rgb::new(0x1e, 0x1e, 0x2e),
        crate::terminal::Rgb::new(0xcd, 0xd6, 0xf4),
    );
    tokens::set_endpoints(ambient);

    assert_eq!(
        black_on_white, mocha,
        "the ambient endpoints reached a themed frame, so the env pair still decides what the \
         ladder is built between"
    );
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
