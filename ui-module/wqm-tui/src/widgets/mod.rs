//! Storyboard widgets. Each module owns one element of the visual language and, behind
//! the `tui-pantry` feature, the preview variants that let it be judged in isolation.
//!
//! Where a widget depicts something the system already has a contract for, it is typed on
//! that contract rather than on a local mock: [`collections`] on N8's name registry,
//! [`daemon_status`] on N49's client seam, [`envelope`] on N12's response envelope. A
//! frame that cannot be built from the real type is a frame depicting a state the system
//! cannot produce.

pub mod collections;
pub mod daemon_status;
pub mod envelope;
pub mod palette_reference;
pub mod palette_sheet;
pub mod store_health;
pub mod tab_bar;
