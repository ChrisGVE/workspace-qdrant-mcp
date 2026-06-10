//! Terminal User Interface (TUI) for interactive wqm operation.
//!
//! Provides scrollable, navigable views for queue inspection,
//! project browsing, and system monitoring.

#[cfg(feature = "tui")]
pub mod app;
#[cfg(feature = "tui")]
pub mod commands;
#[cfg(feature = "tui")]
pub mod event;
#[cfg(feature = "tui")]
pub mod filter;
#[cfg(feature = "tui")]
pub mod search;
#[cfg(feature = "tui")]
pub mod terminal;
#[cfg(feature = "tui")]
pub mod theme;
#[cfg(feature = "tui")]
pub mod util;
#[cfg(feature = "tui")]
pub mod views;

/// Entry point for the TUI. Sets up the terminal, runs the app loop,
/// and restores the terminal on exit.
#[cfg(feature = "tui")]
pub fn run_tui(daemon_addr: String) -> anyhow::Result<()> {
    let mut app = app::App::new(daemon_addr);
    app.run()
}
