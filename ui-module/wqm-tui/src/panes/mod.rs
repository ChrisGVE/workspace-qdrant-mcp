//! Panes — §16's middle tier: **a zone of a screen**.
//!
//! Chris, 20260731: a widget is *"one element of the visual language, including the screen
//! chrome"*; a pane is *"composed sections combining multiple widgets into cohesive panels"* —
//! this crate's reading of which is **a zone**: a list that fills a tab's body, the Service
//! hub's status band, one cell of the Dashboard's 2 × 3 grid.
//!
//! # Why the tier needs a directory of its own
//!
//! *"Panes are the unit of design work from here"*: the Dashboard is six of them, and each is
//! *"detailed individually as panes before being composed into the dashboard"*. Building and
//! judging a cell is the work; composing the screen is what follows. A tier that is the unit of
//! work needs somewhere to put its units.
//!
//! It also makes the tier true in the tree rather than only in a method. `Ingredient::tab()`
//! defaults to `"Widgets"`, so a pane that forgets it lands among the atoms and *looks like*
//! one — no error, no wrong frame, just the wrong idea of what the thing is. `widgets::tier`
//! guards the declaration; the directory is what makes the guard obvious rather than clever.
//!
//! # A pane owns navigation within itself
//!
//! Chris, 20260731, and it is why these are not just functions on a view: *"a pane owns
//! navigation within itself; a view owns which pane has focus; the chrome owns what the status
//! line offers."* So a pane takes [`crate::widgets::chrome::Attention`] and its own zone index
//! — the view says which zone is live, the pane decides what being live looks like for its own
//! heading and its own contents.

pub mod collections;
pub mod config;
pub mod status_band;
pub mod status_block;
pub mod storage;

pub use config::ConfigPane;
pub use status_band::StatusBand;
pub use status_block::StatusBlock;
pub use storage::StorageCell;
