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
use wqm_common::envelope::Severity;
use wqm_tui::capture::capture;
use wqm_tui::terminal;
use wqm_tui::tokens;
use wqm_tui::widgets::{
    collections::Collections,
    daemon_status::DaemonPanel,
    store_health::{StoreHealth, StoreRow},
    tab_bar::TabBar,
    toast::{Toast, ToastDeck, ToastStack},
};

/// One capture: its file stem, the cell grid it is drawn into, and how to draw it.
type Frame = (&'static str, u16, u16, Box<dyn Fn(&mut ratatui::Frame)>);

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
                deck.push(
                    Toast::notice(Severity::Info, "indexed 12 files in projects"),
                    now - Duration::from_millis(900),
                );
                deck.push(
                    Toast::health(tokens::Health::Degraded, "vector store degraded"),
                    now - Duration::from_millis(400),
                );
                deck.push(Toast::error("the write was refused"), now);
                f.render_widget(TabBar::standard(0), Rect::new(0, 0, 62, 1));
                f.render_widget(ToastStack::new(&deck, now), f.area());
            }),
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
