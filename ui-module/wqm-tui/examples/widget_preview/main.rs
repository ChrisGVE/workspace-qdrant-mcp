//! The pantry entry point.
//!
//! The terminal is asked for its background and foreground *here*, before `run!()` takes
//! the screen: `Palette::Derived` interpolates between those two colours, and the query has
//! to happen while the terminal is still in its normal mode and nothing else is competing
//! for its input. Rendering never queries, so a frame costs no round trip.
//!
//! A terminal that does not answer — or a piped stdout, which is every `cargo pantry dump`
//! — leaves the fallback endpoints in place, so derived frames still render; they render
//! the black-to-white ladder r02 was authored against rather than this terminal's.
//! `WQM_TUI_TERM_BG` / `WQM_TUI_TERM_FG` override both, which is how a derived frame is
//! captured reproducibly with no terminal to ask.

fn main() -> std::io::Result<()> {
    if let Some(endpoints) = wqm_tui::terminal::detect() {
        wqm_tui::tokens::set_endpoints(endpoints);
    }
    tui_pantry::run!()
}
