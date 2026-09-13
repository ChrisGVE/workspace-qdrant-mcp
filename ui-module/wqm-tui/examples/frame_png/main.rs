//! Writes the storyboard's frames to PNG files, with no terminal involved.
//!
//! ```bash
//! cargo run -p wqm-tui --features png-capture --example frame_png -- out/
//! ```
//!
//! # `WQM_TUI_TERM_BG` / `WQM_TUI_TERM_FG` no longer change a single frame, and that is new
//!
//! They used to be the load-bearing thing here. `Palette::Derived` interpolates the neutral
//! ladder between the terminal's own background and foreground, this process has no terminal
//! to ask, and without the pair every rung was generated against the black-to-white fallback —
//! so leaving them off produced a valid frame of the wrong ladder.
//!
//! That stopped being true when this example started stating its own theme (below). Under
//! `Palette::Bundled` the ladder is built between the THEME's background and foreground
//! (`tokens::ladder_endpoints`), and `capture` paints its ground from the same place, so
//! nothing in the render path consults the ambient endpoints at all. **Measured: every frame
//! this example writes is byte-identical with the pair set and with it unset.**
//!
//! Keeping the theme here rather than the env pair is the better of the two anyway — a
//! reproducible capture should not depend on a variable the person running it has to remember,
//! and a frame of *the wrong ladder* is precisely the failure that cannot be seen by looking
//! at the frame.
//!
//! Read the output for layout, weight, glyph and neutral fidelity. Not for hue —
//! `wqm_tui::capture`'s module docs give the measured reason.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Layout, Rect};
use wqm_tui::capture::capture;
use wqm_tui::panes::collections::Collections;
use wqm_tui::terminal;
use wqm_tui::tokens;
use wqm_tui::widgets::{
    config_table::{ConfigTable, Edit, Entry, Focus, Row, UNSET},
    daemon_status::DaemonPanel,
    modal::{Fill, Modal},
    store_health::{StoreHealth, StoreRow},
    surface::{ConditionBand, Surface},
    tab_bar::TabBar,
    toast::{Toast, ToastDeck, ToastStack},
};

/// One capture: its file stem, the cell grid it is drawn into, and how to draw it.
type Frame = (&'static str, u16, u16, Box<dyn Fn(&mut ratatui::Frame)>);

/// The gallery enumerates `ratatui-themes`' own closed set, so it exists only when that
/// optional dependency does.
///
/// This example is gated on `png-capture` alone, and it used to name `ThemeGallery`
/// unconditionally — so `--features png-capture` on its own did not compile. A feature that
/// is required must be declared or be optional in the code; it cannot be assumed.
#[cfg(feature = "themes-preview")]
fn gallery_frames() -> Vec<Frame> {
    vec![(
        "theme-gallery",
        112,
        22,
        Box::new(|f: &mut ratatui::Frame| {
            f.render_widget(wqm_tui::styles::theme_sheet::ThemeGallery, f.area())
        }),
    )]
}

#[cfg(not(feature = "themes-preview"))]
fn gallery_frames() -> Vec<Frame> {
    Vec::new()
}

/// A whole screen under a stated condition — the Service tab, with the stores below it.
///
/// Under `DaemonUnreachable` the component readings are drawn as **unreadable** rather than
/// healthy: seen through a dead daemon, every one of them is unknown, and four green dots
/// under an unreachable banner would be a frame of a state the system cannot produce.
fn unreachable_screen(f: &mut ratatui::Frame, condition: tokens::Condition) {
    let area = f.area();
    f.render_widget(Surface::with_condition(condition), area);
    let [body, band] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(Surface::reserved_rows(condition)),
    ])
    .areas(area);
    let [tabs, _gap, stores] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(body);
    f.render_widget(TabBar::standard(3), tabs);
    match condition {
        tokens::Condition::Nominal => f.render_widget(StoreHealth::nominal(), stores),
        tokens::Condition::DaemonUnreachable => f.render_widget(
            StoreHealth::new(vec![
                StoreRow::unreadable("daemon"),
                StoreRow::unreadable("vector"),
                StoreRow::unreadable("graph"),
                StoreRow::unreadable("relational"),
            ]),
            stores,
        ),
    }
    // Last, over nothing.
    f.render_widget(ConditionBand::with_condition(condition), band);
}

/// The config screen's keys, as the r06 frame spells them. One producer for every
/// config-table capture below, so three frames cannot disagree about what is on screen.
fn config_rows() -> Vec<Row> {
    vec![
        Row::Group("qdrant".into()),
        Row::Entry(Entry::new(
            "URL",
            "http://localhost:6333",
            "http://localhost:6333",
        )),
        Row::Entry(Entry::new("API key", UNSET, UNSET)),
        Row::Group("watcher".into()),
        Row::Entry(Entry::new("Debounce [ms]", "2000", "1500")),
    ]
}

/// The entry index of `watcher / Debounce [ms]` — the key the frame edits.
const DEBOUNCE: usize = 2;

/// The storyboard's screen, and the floor Chris named. Every modal-framework frame is a WHOLE
/// screen at one of these two: a window judged on an empty buffer is judged against nothing,
/// and *"is this window washed out and sad"* only has an answer over the page it covers.
const SCREEN: (u16, u16) = (125, 34);
const FLOOR: (u16, u16) = (100, 30);

/// The modal framework's frames — round 1's deliverable.
///
/// The storyboard IS the product: a feature whose frame cannot be produced is not done, which
/// is why these are generated from the same `Ingredient::render` the pantry browses rather
/// than from a second path that could drift from it.
fn modal_framework_frames() -> Vec<Frame> {
    use wqm_tui::views::modal_framework::frames as mf;
    use wqm_tui::views::modal_framework::record::{CursorExtent, Mode, Reference, Scheme};
    use wqm_tui::widgets::edit_field::Edit;
    use wqm_tui::widgets::modal_frame::Footprint;

    /// The gate's proposal, in force for every frame but the brackets.
    fn proposed(draw: impl FnOnce()) {
        mf::with_tint(tokens::ModalTint::Accent, mf::PROPOSED_WASH, draw);
    }

    fn editing() -> Mode {
        Mode::Edit {
            at: 6,
            edit: Some(Edit::insert("256")),
        }
    }

    let mut frames: Vec<Frame> = Vec::new();
    let mut add = |name: &'static str, size: (u16, u16), draw: Box<dyn Fn(&mut ratatui::Frame)>| {
        frames.push((name, size.0, size.1, draw));
    };

    add(
        "mf-01-record-view",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame::default().draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-02-record-edit",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame::view(editing()).draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-03-record-scrolled",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    offset: 5,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-04-record-empty",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    empty: true,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-05-table-in-modal",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::table_frame(area, f.buffer_mut(), false);
            });
        }),
    );
    add(
        "mf-06-table-empty",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::empty_table_frame(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-07-slide-t00",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| mf::slide_frame(area, f.buffer_mut(), 0.0));
        }),
    );
    add(
        "mf-07-slide-t05",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| mf::slide_frame(area, f.buffer_mut(), 0.5));
        }),
    );
    add(
        "mf-07-slide-t10",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| mf::slide_frame(area, f.buffer_mut(), 1.0));
        }),
    );
    add(
        "mf-08-confirm-over-window",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::confirm_frame(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-09-dropdown-open",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::dropdown_frame(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-10-help",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::help_frame(area, f.buffer_mut());
            });
        }),
    );

    // The A/B pairs the gate keeps for Chris's look.
    add(
        "mf-11-footprint-A-literal",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    footprint: Footprint::HelpDerived,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-11-footprint-B-framework",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame::default().draw(area, f.buffer_mut());
            });
        }),
    );
    for (name, tint, strength) in [
        ("mf-12-tint-neutral", tokens::ModalTint::Neutral, 0.28f32),
        ("mf-12-tint-blue-014", tokens::ModalTint::Accent, 0.14),
        ("mf-12-tint-blue-028", tokens::ModalTint::Accent, 0.28),
        ("mf-12-tint-blue-040", tokens::ModalTint::Accent, 0.40),
        ("mf-12-tint-lavender", tokens::ModalTint::Selected, 0.28),
    ] {
        add(
            name,
            SCREEN,
            Box::new(move |f: &mut ratatui::Frame| {
                let area = f.area();
                mf::with_tint(tint, strength, || {
                    mf::RecordFrame::default().draw(area, f.buffer_mut());
                });
            }),
        );
    }
    add(
        "mf-12-fallback-neutral-window-washed-form",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            mf::with_tint(tokens::ModalTint::Neutral, mf::PROPOSED_WASH, || {
                mf::RecordFrame {
                    mode: editing(),
                    scheme: Scheme::AccentWash,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    // The four frames that retire arm A. Two themes, two arms, so the pair can be read the
    // only way it settles anything: the harness's own Mocha is among the ROOMIEST of the
    // fifteen ladders, so A looks survivable there — which is why it was proposed — and the
    // thing to look at is the same pair on the thinnest one.
    for (name, theme, arm_a) in [
        ("mf-13-edit-fields-A-mocha", None, true),
        (
            "mf-13-edit-fields-A-solarized-dark",
            Some(mf::ADVERSARIAL_THEME),
            true,
        ),
        ("mf-13-edit-fields-Aplus-mocha", None, false),
        (
            "mf-13-edit-fields-Aplus-solarized-dark",
            Some(mf::ADVERSARIAL_THEME),
            false,
        ),
    ] {
        add(
            name,
            SCREEN,
            Box::new(move |f: &mut ratatui::Frame| {
                let area = f.area();
                let buf = f.buffer_mut();
                // The frame, under whichever theme this entry names. Written twice rather
                // than hoisted into a closure: the closure would capture `buf` mutably, which
                // makes it `FnMut` and no longer something `with_theme`'s `FnOnce` can take.
                match theme {
                    Some(theme) => mf::with_theme(theme, || {
                        proposed(|| {
                            mf::RecordFrame {
                                mode: editing(),
                                arm_a,
                                ..mf::RecordFrame::default()
                            }
                            .draw(area, buf);
                        });
                    }),
                    None => proposed(|| {
                        mf::RecordFrame {
                            mode: editing(),
                            arm_a,
                            ..mf::RecordFrame::default()
                        }
                        .draw(area, buf);
                    }),
                }
            }),
        );
    }
    add(
        "mf-13-edit-fields-C",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    mode: editing(),
                    scheme: Scheme::SelectionDerived,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-14-third-column-A-text",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    reference: Reference::Text("DEFAULT"),
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-14-third-column-B-band",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame::default().draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-14-third-column-none",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    reference: Reference::None,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-15-cursor-band-A-full-row",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    mode: Mode::View { at: 6 },
                    cursor_extent: CursorExtent::FullRow,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-15-cursor-band-B-to-value",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame {
                    mode: Mode::View { at: 6 },
                    cursor_extent: CursorExtent::ToValue,
                    ..mf::RecordFrame::default()
                }
                .draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-16-table-pinned-shown",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::table_frame(area, f.buffer_mut(), true);
            });
        }),
    );
    add(
        "mf-16-table-pinned-dropped",
        SCREEN,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::table_frame(area, f.buffer_mut(), false);
            });
        }),
    );

    // The floor, where every argument about room has to hold too.
    add(
        "mf-17-floor-record",
        FLOOR,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::RecordFrame::default().draw(area, f.buffer_mut());
            });
        }),
    );
    add(
        "mf-17-floor-table",
        FLOOR,
        Box::new(|f: &mut ratatui::Frame| {
            let area = f.area();
            proposed(|| {
                mf::table_frame(area, f.buffer_mut(), false);
            });
        }),
    );

    frames
}

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".to_string()));
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("cannot write to {}: {e}", out.display());
        std::process::exit(1);
    }

    // Same detection path the TUI uses, so a capture and a live frame are generated from
    // identical endpoints when the env pair is set.
    match terminal::detect() {
        Some(endpoints) => {
            eprintln!(
                "endpoints: bg #{:02x}{:02x}{:02x} fg #{:02x}{:02x}{:02x}",
                endpoints.background.r,
                endpoints.background.g,
                endpoints.background.b,
                endpoints.foreground.r,
                endpoints.foreground.g,
                endpoints.foreground.b,
            );
            tokens::set_endpoints(endpoints);
        }
        None => eprintln!(
            "no endpoints — falling back to black/white. Set WQM_TUI_TERM_BG and \
             WQM_TUI_TERM_FG to capture against a real theme."
        ),
    }

    // **The bundled theme has to be in force, or every hue falls back to an ANSI slot.**
    //
    // `tokens::active_theme` answers `None` for every source but `Palette::Bundled` — a theme
    // that has been *chosen* is not a theme that is *in force* — so without these two lines
    // `modal_border()` resolves to `Color::Magenta`, `Rgb::from_color` cannot read a slot, and
    // `modal_fill` hands back the bare layer. Measured: the five tint frames came out
    // byte-identical, and the existing modal frames grew a `#ff00ff` border. A capture that
    // silently depicted the wrong palette is exactly the artifact `capture`'s module docs
    // exist to stop producing, so the theme is stated here rather than left to whatever the
    // process happened to be in.
    //
    // Catppuccin Mocha because that is what the harness paints with, what the tests pin, and
    // what §15 made the shipping default. `Bundled` is an RGB source, so `capture`'s own
    // RGB-only guard leaves it alone rather than overriding it with `Derived`.
    tokens::Palette::set(tokens::Palette::Bundled);
    tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    let frames: Vec<Frame> = vec![
        (
            "tab-bar",
            62,
            1,
            Box::new(|f: &mut ratatui::Frame| f.render_widget(TabBar::standard(0), f.area())),
        ),
        (
            "collections",
            48,
            5,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(Collections::new(Some(0)).with_reserved(), f.area())
            }),
        ),
        (
            "daemon-status",
            62,
            4,
            Box::new(|f: &mut ratatui::Frame| f.render_widget(DaemonPanel::nominal(), f.area())),
        ),
        (
            "store-health",
            44,
            6,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    StoreHealth::new(vec![
                        StoreRow::bound("daemon", "memexd", tokens::Health::Healthy),
                        StoreRow::bound("vector", "qdrant", tokens::Health::Healthy),
                        StoreRow::unbound("queue_processor", tokens::Health::Degraded),
                        StoreRow::unreadable("watch_supervisor"),
                    ]),
                    f.area(),
                )
            }),
        ),
        // The composition is where flatness becomes judgeable (handover §10 item 4), and it
        // is the frame a grid dump is least able to settle.
        (
            "service-zone",
            62,
            12,
            Box::new(|f: &mut ratatui::Frame| {
                let [tabs, daemon, stores] = Layout::vertical([
                    Constraint::Length(2),
                    Constraint::Length(4),
                    Constraint::Min(0),
                ])
                .areas(f.area());
                f.render_widget(TabBar::standard(3), tabs);
                f.render_widget(DaemonPanel::nominal(), daemon);
                f.render_widget(StoreHealth::nominal(), stores);
            }),
        ),
        // The A/B Chris has to judge: the same screen, nominal and unreachable. A grid dump
        // cannot show a background wash at all, so this pair is the only honest instrument for
        // it — read for the wash's weight against the content, not for its exact hue.
        (
            "surface-nominal",
            62,
            10,
            Box::new(|f: &mut ratatui::Frame| unreachable_screen(f, tokens::Condition::Nominal)),
        ),
        (
            "surface-daemon-unreachable",
            62,
            10,
            Box::new(|f: &mut ratatui::Frame| {
                unreachable_screen(f, tokens::Condition::DaemonUnreachable)
            }),
        ),
        // The tolerance, side by side. A capture is not honest about hue, so this is the
        // *relative* instrument — the ANSI dump of `Surface / Wash Strengths` in a real
        // terminal is what settles the value.
        (
            "wash-strengths",
            74,
            5,
            Box::new(|f: &mut ratatui::Frame| {
                let area = f.area();
                for (i, mix) in [0.06f32, 0.10, 0.14, 0.18, 0.22].iter().enumerate() {
                    let strip = Rect::new(area.x, area.y + i as u16, area.width, 1);
                    if let Some(colour) =
                        tokens::wash_at(tokens::Condition::DaemonUnreachable, *mix)
                    {
                        f.buffer_mut()
                            .set_style(strip, ratatui::style::Style::default().bg(colour));
                    }
                    let current = if (*mix - tokens::WASH_MIX).abs() < f32::EPSILON {
                        " ← current"
                    } else {
                        ""
                    };
                    f.render_widget(
                        ratatui::widgets::Paragraph::new(ratatui::text::Line::from(vec![
                            ratatui::text::Span::styled(
                                format!(" {mix:.2}  "),
                                tokens::strong_style(),
                            ),
                            ratatui::text::Span::styled(
                                "normal body text  ",
                                tokens::normal_style(),
                            ),
                            ratatui::text::Span::styled("muted label  ", tokens::muted_style()),
                            ratatui::text::Span::styled("faint default  ", tokens::faint_style()),
                            ratatui::text::Span::styled(
                                tokens::Health::Offline.glyph(),
                                ratatui::style::Style::default()
                                    .fg(tokens::Health::Offline.color()),
                            ),
                            ratatui::text::Span::styled(current, tokens::muted_style()),
                        ])),
                        strip,
                    );
                }
            }),
        ),
        // The toast is the one element whose *placement* is the design (lower-right, one clear
        // cell from each edge), and placement is exactly what a grid dump of a widget-sized area
        // cannot show. Captured screen-sized for that reason.
        (
            "toast-stack",
            62,
            14,
            Box::new(|f: &mut ratatui::Frame| {
                let now = Instant::now();
                let mut deck = ToastDeck::new();
                // Pushed at stated past moments — the caller owns the clock, which is what lets
                // a capture show a toast mid-life without waiting for one.
                let transition = |from, to, message: &str| {
                    Toast::transition(from, to, message).expect("a capture must show a change")
                };
                deck.push(
                    transition(
                        tokens::Health::Healthy,
                        tokens::Health::Offline,
                        "vector store offline — qdrant unreachable",
                    ),
                    now - Duration::from_millis(900),
                );
                deck.push(
                    transition(
                        tokens::Health::Offline,
                        tokens::Health::Degraded,
                        "vector store degraded — rebuilding",
                    ),
                    now - Duration::from_millis(400),
                );
                deck.push(
                    transition(
                        tokens::Health::Degraded,
                        tokens::Health::Healthy,
                        "vector store recovered",
                    ),
                    now,
                );
                f.render_widget(TabBar::standard(0), Rect::new(0, 0, 62, 1));
                f.render_widget(ToastStack::new(&deck, now), f.area());
            }),
        ),
        (
            "config-cursor",
            70,
            6,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    ConfigTable::new(config_rows()).focus(Focus::Cursor(DEBOUNCE)),
                    f.area(),
                )
            }),
        ),
        (
            "config-editing-insert",
            70,
            6,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    ConfigTable::new(config_rows())
                        .focus(Focus::Editing(DEBOUNCE, Edit::insert("2000"))),
                    f.area(),
                )
            }),
        ),
        (
            "config-editing-normal",
            70,
            6,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    ConfigTable::new(config_rows())
                        .focus(Focus::Editing(DEBOUNCE, Edit::normal("2000", 1))),
                    f.area(),
                )
            }),
        ),
        (
            "modal-layer1",
            70,
            12,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    Modal::new(
                        "Discard changes?",
                        "watcher.debounce_ms has been edited and not saved.",
                    )
                    .action("↵", "discard")
                    .action("Esc", "keep editing"),
                    f.area(),
                )
            }),
        ),
        (
            // The one frame that judges all three against each other: a modal, a modal on
            // top of it, and a toast over both. If the stack does not read as three
            // surfaces here, the depth model has failed regardless of what the tokens say.
            "modal-stack-with-toast",
            70,
            14,
            Box::new(|f: &mut ratatui::Frame| {
                let now = Instant::now();
                let mut deck = ToastDeck::new();
                deck.push(
                    Toast::transition(
                        tokens::Health::Healthy,
                        tokens::Health::Offline,
                        "vector store offline",
                    )
                    .expect("a change of state"),
                    now,
                );
                let area = f.area();
                f.render_widget(
                    Modal::with_body(
                        "Discard changes?",
                        vec!["watcher.debounce_ms has been edited and not saved.".into()],
                    )
                    .action("↵", "discard")
                    .action("Esc", "keep editing"),
                    Rect {
                        y: area.y + 1,
                        height: area.height - 4,
                        ..area
                    },
                );
                f.render_widget(
                    Modal::new("Really discard?", "This cannot be undone.")
                        .fill(Fill::Layer2)
                        .action("y", "yes")
                        .action("n", "no"),
                    area,
                );
                f.render_widget(ToastStack::new(&deck, now), area);
            }),
        ),
    ];
    let frames: Vec<Frame> = frames
        .into_iter()
        .chain(gallery_frames())
        .chain(modal_framework_frames())
        .collect();

    for (name, cols, rows, draw) in frames {
        let png = match capture(cols, rows, |f| draw(f)) {
            Ok(png) => png,
            Err(e) => {
                eprintln!("{name}: {e}");
                std::process::exit(1);
            }
        };
        let path = out.join(format!("{name}.png"));
        if let Err(e) = std::fs::write(&path, &png) {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
        println!("{} ({} bytes)", path.display(), png.len());
    }
}
