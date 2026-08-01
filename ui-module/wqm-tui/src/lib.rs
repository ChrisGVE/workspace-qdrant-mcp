//! wqm TUI storyboard widgets (P01-GT001).
//!
//! Every widget here renders against VISUAL-LANGUAGE.md r02. A widget that invents an
//! ad-hoc highlight is a defect — reach for [`tokens`] instead of a colour literal.
//!
//! The crate is a design instrument: it exists so storyboard frames are produced by the
//! real ratatui renderer rather than approximated, and so each frame can be browsed and
//! compared in `cargo pantry`.

#[cfg(feature = "png-capture")]
pub mod capture;
pub mod encoding;
pub mod health;
pub mod names;
pub mod panes;
pub mod styles;
pub mod terminal;
pub mod tokens;
pub mod views;
pub mod widgets;

/// Serialises every test that touches process-global rendering state.
///
/// The active palette ([`tokens::Palette`]) and the active encoding
/// ([`encoding::Encoding`]) are process-global by design — a widget deep in a render tree
/// must be able to ask what it is rendering into without being handed a context. The cost is
/// that a test which sets one of them, renders, and then asserts is racing every other test
/// that does the same, and cargo runs them in parallel by default.
///
/// One lock for the whole crate rather than one per module: two independent locks around the
/// same global is how a lock-ordering deadlock is built.
#[cfg(test)]
pub(crate) fn global_state_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panicking test poisons the lock but leaves no invariant broken — every holder either
    // restores what it found or is followed by one that sets the value it needs. Recovering
    // keeps one failure from cascading into every later test as a poison error.
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
