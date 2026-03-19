//! Terminal lifecycle management.
//!
//! Handles raw mode, alternate screen, and ensures cleanup on panic.

use std::io::{self, Stdout};

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

/// Initialize the terminal for TUI rendering.
///
/// Enables raw mode, enters the alternate screen, enables mouse capture,
/// and installs a panic hook that restores the terminal before printing
/// the panic message.
pub fn init() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    install_panic_hook();
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

/// Restore the terminal to its original state.
///
/// Leaves the alternate screen, disables mouse capture, and disables raw mode.
/// This is safe to call multiple times.
pub fn restore() -> io::Result<()> {
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}

/// Install a panic hook that restores the terminal before the default
/// panic handler runs, so the user sees the panic message on a normal
/// terminal rather than a garbled raw-mode screen.
fn install_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = restore();
        original_hook(panic_info);
    }));
}
