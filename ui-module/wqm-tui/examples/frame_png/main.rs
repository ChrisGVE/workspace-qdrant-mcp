//! Writes the storyboard's frames to PNG files, with no terminal involved.
//!
//! ```bash
//! cargo run -p wqm-tui --features png-capture --example frame_png -- out/
//! WQM_TUI_TERM_BG=#1e1e2e WQM_TUI_TERM_FG=#cdd6f4 \
//!   cargo run -p wqm-tui --features png-capture --example frame_png -- out/
//! ```
//!
//! The env pair matters more here than anywhere else in the crate. `Palette::Derived`
//! interpolates between the terminal's own background and foreground, and this process has
//! no terminal to ask — so without them every neutral is generated against the black-to-white
//! fallback rather than against the theme the storyboard is judged on. Setting them is how a
//! capture becomes reproducible *and* representative; leaving them off produces a valid frame
//! of the wrong ladder.
//!
//! Read the output for layout, weight, glyph and neutral fidelity. Not for hue —
//! `wqm_tui::capture`'s module docs give the measured reason.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Layout, Rect};
use wqm_tui::capture::capture;
use wqm_tui::terminal;
use wqm_tui::tokens;
use wqm_tui::widgets::{
    collections::Collections,
    config_table::{ConfigTable, Edit, Entry, Focus, Row, UNSET},
    modal::{Fill, Modal},
    theme_sheet::{semantic, ThemeGallery, ThemeSheet},
    daemon_status::DaemonPanel,
    store_health::{StoreHealth, StoreRow},
    surface::{ConditionBand, Surface},
    tab_bar::TabBar,
    toast::{Toast, ToastDeck, ToastStack},
};

/// One capture: its file stem, the cell grid it is drawn into, and how to draw it.
type Frame = (&'static str, u16, u16, Box<dyn Fn(&mut ratatui::Frame)>);

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
        (
            // Honest for hue, unusually: every swatch here is a literal Color::Rgb from
            // theme data, which is the one thing this backend resolves byte-exactly (§12).
            "theme-catppuccin-mocha",
            112,
            34,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    ThemeSheet::new(wqm_tui::themes::catppuccin_mocha())
                        .with_semantic(semantic::CATPPUCCIN_MOCHA),
                    f.area(),
                )
            }),
        ),
        (
            "theme-catppuccin-latte",
            112,
            34,
            Box::new(|f: &mut ratatui::Frame| {
                f.render_widget(
                    ThemeSheet::new(wqm_tui::themes::catppuccin_latte()),
                    f.area(),
                )
            }),
        ),
        (
            "theme-gallery",
            112,
            22,
            Box::new(|f: &mut ratatui::Frame| f.render_widget(ThemeGallery, f.area())),
        ),
    ];

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
