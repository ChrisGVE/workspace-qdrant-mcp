//! wqm TUI storyboard widgets (P01-GT001).
//!
//! Every widget here renders against VISUAL-LANGUAGE.md r02. A widget that invents an
//! ad-hoc highlight is a defect — reach for [`tokens`] instead of a colour literal.
//!
//! The crate is a design instrument: it exists so storyboard frames are produced by the
//! real ratatui renderer rather than approximated, and so each frame can be browsed and
//! compared in `cargo pantry`.

pub mod names;
pub mod terminal;
pub mod tokens;
pub mod widgets;
