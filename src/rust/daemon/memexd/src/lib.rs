//! memexd library facade.
//!
//! The daemon entry-point lives in `src/main.rs`. This library target
//! re-exports a narrow subset of internal modules so integration tests
//! under `tests/` can exercise them without re-implementing parallel
//! copies. Production binary code does NOT consume this crate as a
//! library — main.rs uses the modules directly.
//!
//! Public surface (intentionally minimal):
//! * [`control_port`] — cross-process single-instance lock primitive
//!   (spec 16 §10.1).

pub mod control_port;
