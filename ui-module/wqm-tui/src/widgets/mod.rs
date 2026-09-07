//! Storyboard widgets. Each module owns one element of the visual language and, behind
//! the `tui-pantry` feature, the preview variants that let it be judged in isolation.
//!
//! Where a widget depicts something the system already has a contract for, it is typed on
//! that contract rather than on a local mock: [`daemon_status`] on N49's client seam,
//! [`envelope`] on N12's response envelope. A frame that cannot be built from the real type is
//! a frame depicting a state the system cannot produce.
//!
//! # Only widgets are here
//!
//! A zone lives in [`crate::panes`], a screen in [`crate::views`], the vocabulary and the
//! instruments in [`crate::styles`]. `Ingredient::tab()` defaults to `"Widgets"`, so a module
//! filed in the wrong directory used to be one forgotten method away from *looking like* an
//! atom; now the directory says the tier and the method agrees with it. `crate::tier` guards
//! that they agree.
//!
//! [`chrome`] is the exception to "a widget is a zone's content": §16's composition model
//! (Chris, 20260731) counts the screen furniture — title bar, status line, rules, selectors,
//! headings — as widgets too, and it lives there as one family.

pub mod chrome;
pub mod config_table;
pub mod daemon_status;
pub mod edit_field;
pub mod envelope;
pub mod modal;
pub mod store_health;
pub mod surface;
pub mod tab_bar;
pub mod toast;
